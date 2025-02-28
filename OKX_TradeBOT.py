
# 替换为你的 API Key 和 Secret
api_key = 'api_key'
api_secret = 'secret'
passphrase = 'password'
import time
import pandas as pd
import numpy as np
import joblib  # 新增模型保存库
from okx import MarketData, Trade, Account, exceptions
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime

# 配置参数
symbol = 'BTC-USDT-SWAP'
timeframe = '1m'
model_path = 'xgboost_model.joblib'  # 模型保存路径
initial_data_size = 20000  # 从1440增加到20000
PREDICTION_WINDOW = 5

# 初始化API客户端
market_data_api = MarketData.MarketAPI(api_key, api_secret, passphrase, flag='0', debug=False)
trade_api = Trade.TradeAPI(api_key, api_secret, passphrase,flag='0', debug=False)
account_api = Account.AccountAPI(api_key, api_secret, passphrase,flag='0', debug=False)

# 全局变量
historical_data = pd.DataFrame()
model = None


# 初始化历史数据（获取20000根1分钟K线）

def get_current_position():
    """获取当前合约持仓方向及数量"""
    positions = account_api.get_positions(instType='SWAP')
    if positions['code'] != '0':
        print("获取持仓失败:", positions)
        return None, 0

    for pos in positions['data']:
        if pos['instId'] == symbol and pos['posSide'] in ['long', 'short']:
            return pos['posSide'], float(pos['pos'])
    return None, 0  # 无持仓
# 模型训练与保存
def train_and_save_model(historical_data):
    global model
    try:
        features = ['ma_20', 'ma_50', 'rsi', 'bollinger_upper',
                    'bollinger_lower', 'momentum', 'volatility',
                    'returns_lag_1', 'returns_lag_3', 'returns_lag_5']

        if len(historical_data) < 1000:  # 至少需要10000条有效数据
            print(historical_data)
            raise Exception("可用训练数据不足")

        print("\n----- 开始模型训练 -----")
        print(f"训练数据量: {len(historical_data)} 条")

        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=3)

        # 参数网格
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }

        # 网格搜索
        grid_search = GridSearchCV(
            XGBClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring='accuracy',
            verbose=3  # 显示详细训练过程
        )

        grid_search.fit(historical_data[features], historical_data['target'])

        # 保存最佳模型
        model = grid_search.best_estimator_
        joblib.dump(model, model_path)
        print(f"模型已保存至 {model_path}")

        # 输出最佳参数
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"验证集准确率: {grid_search.best_score_:.2f}")

    except Exception as e:
        print(f"模型训练失败: {str(e)}")
        raise


# 加载已有模型
def load_model():
    global model
    try:
        model = joblib.load(model_path)
        print(f"已加载预训练模型: {model_path}")
        return True
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False


def execute_trade():
        if model is None:

            return
        if len(historical_data1) < 100:
            print(f"数据不足 ({len(historical_data1)} < 100)，跳过交易")
            return

        features = ['ma_20', 'ma_50', 'rsi', 'bollinger_upper',
                    'bollinger_lower', 'momentum', 'volatility',
                    'returns_lag_1', 'returns_lag_3', 'returns_lag_5']

        # 获取最新特征
        latest_data = historical_data1.iloc[[-1]][features]
        print(historical_data1.iloc[[-1]])

        # 获取实时账户信息
        balance_info = account_api.get_account_balance(ccy='USDT')

        if balance_info['code'] != '0':
            print(balance_info)
            return

        avail_balance = float(balance_info['data'][0]['details'][0]['availBal'])
        if avail_balance < 10:
            print("余额不足10 USDT")
            return

        # 风险控制（每次最多5%资金）
        risk_percent = 0.05
        position_size = avail_balance * risk_percent

        # 获取最新价格
        ticker = market_data_api.get_ticker(instId=symbol)
        if ticker['code'] != '0':
            print("获取价格失败")
            return

        last_price = float(ticker['data'][0]['last'])

        contract_size =0.1  # 永续合约面值0.01 BTC


        # 生成预测

        probas = model.predict_proba(latest_data)[0]
        confidence_threshold = 0.65  # 置信度阈值

        if probas[1] > confidence_threshold:
            print(f"强力做多信号（置信度{probas[1]:.2%}）")
            current_side, current_size = get_current_position()

            # 已有空头持仓需要先平仓
            if current_side == 'short':
                print(f"平空头仓位（数量：{current_size}）")
                close_order = trade_api.place_order(
                    instId=symbol,
                    tdMode='cross',
                    side='buy',  # 平空方向为买入
                    posSide='short',
                    ordType='market',
                    sz=str(current_size)  # 平全部空仓
                )
                print("平仓结果:", close_order)

            # 开多头仓位（只有无持仓或已平仓后操作）
            if current_side != 'short':
                order = trade_api.place_order(
                    instId=symbol,
                    tdMode='cross',
                    side='buy',
                    posSide='long',
                    ordType='market',
                    sz=str(round(contract_size, 3))
                )
                print("开多结果:", order)

        elif probas[0] > confidence_threshold:
                print(f"强力做空信号（置信度{probas[0]:.2%}）")
                current_side, current_size = get_current_position()

                # 已有多头持仓需要先平仓
                if current_side == 'long':
                    print(f"平多头仓位（数量：{current_size}）")
                close_order = trade_api.place_order(
                    instId=symbol,
                    tdMode='cross',
                    side='sell',  # 平多方向为卖出
                    posSide='long',
                    ordType='market',
                    sz=str(current_size)  # 平全部多仓
                )
                print("平仓结果:", close_order)

                # 开空头仓位（只有无持仓或已平仓后操作）
                if current_side != 'long':
                    order = trade_api.place_order(
                        instId=symbol,
                        tdMode='cross',
                        side='sell',
                        posSide='short',
                        ordType='market',
                        sz=str(round(contract_size, 3)))
                    print("开空结果:", order)
        else:
            print("未达到置信度阈值，放弃交易")

        # if order['code'] != '0':
        #     print(f"下单失败: {order}")



def compute_bollinger_bands(close, window=20, k=2):
    """
    计算布林带（Bollinger Bands）
    :param close: pandas Series，收盘价数据
    :param window: int，移动平均窗口大小，默认为 20
    :param k: int，标准差的倍数，默认为 2
    :return: (pandas Series, pandas Series)，上轨线和下轨线
    """
    middle_band = close.rolling(window=window).mean()  # 中轨线
    std_dev = close.rolling(window=window).std()  # 标准差
    upper_band = middle_band + k * std_dev  # 上轨线
    lower_band = middle_band - k * std_dev  # 下轨线
    return upper_band, lower_band

# 计算 RSI
def compute_rsi(close, period=14):
    """
    计算 RSI（相对强弱指数）
    :param close: pandas Series，收盘价数据
    :param period: int，RSI 的计算周期，默认为 14
    :return: pandas Series，RSI 值
    """
    delta = close.diff()  # 计算价格变化
    gain = delta.where(delta > 0, 0)  # 上涨幅度
    loss = -delta.where(delta < 0, 0)  # 下跌幅度

    # 计算平均上涨和下跌幅度（指数移动平均）
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    # 计算相对强度（RS）和 RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_target(series, window=5, threshold=0.001):
    future_prices = series.shift(-window)
    price_change = (future_prices - series) / series
    # 仅当价格变化超过阈值时才视为有效信号
    return np.where(price_change > threshold, 1,
                   np.where(price_change < -threshold, 0, np.nan))


def initialize_historical_data():
    print(f"开始获取初始{initial_data_size}条数据...")

    all_data = []
    remaining = initial_data_size
    retry_count = 0
    max_retries = 5
    last_timestamp = None  # 用于记录上一次请求的最后一条数据的时间戳

    while remaining > 0 and retry_count < max_retries:
        limit = min(remaining, 300)  # 每次最多请求300条数据
        try:
            # 构造请求参数
            params = {
                'instId': symbol,
                'bar': timeframe,
                'limit': limit
            }
            # 如果不是第一次请求，添加 before 参数
            if last_timestamp is not None:
                params['after'] = str(last_timestamp)

            response = market_data_api.get_history_candlesticks(**params)

            if response['code'] == '0':
                new_data = response['data']
                if not new_data:  # 如果没有新数据，退出循环
                    break

                # 将新数据添加到总数据中
                all_data.extend(new_data)
                remaining -= len(new_data)
                print(f"已获取 {len(all_data)}/{remaining} 条数据")
                print(new_data)

                # 更新 last_timestamp 为本次请求的最后一条数据的时间戳
                last_timestamp = int(new_data[-1][0])  # 时间戳在第0列
                retry_count = 0  # 重置重试计数器
            else:
                print(f"错误: {response['msg']}")
                retry_count += 1
                time.sleep(2)
        except Exception as e:
            print(f"请求异常: {str(e)}")
            retry_count += 1
            time.sleep(5)

    # 处理数据
    if len(all_data) == 0:
        raise Exception("无法获取初始数据")

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'volCcy', 'volCcyQuote', 'confirm'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[
        ['open', 'high', 'low', 'close', 'volume']
    ].astype(float)

    # 处理特征（允许部分NaN）
    print(f"有效数据量: {len(df)} 条")

    historical_data = process_features(df)
    print(f"初始化完成 | 有效数据量: {len(historical_data)} 条")
    return historical_data

def process_features(df):
    if df.empty:
        return df

    # ========== 关键修复1：确保数据时间顺序 ==========
    # 将索引转换为DatetimeIndex（如果尚未转换）
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 按时间升序排列（确保滚动计算方向正确）
    df = df.sort_index(ascending=True)

    # ========== 关键修复2：滚动窗口参数修正 ==========
    # 计算收益率（无未来数据）
    df['returns'] = df['close'].pct_change()

    # 移动平均（严格使用完整窗口）
    df['ma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
    df['ma_50'] = df['close'].rolling(window=50, min_periods=50).mean()

    # 计算RSI（避免前19个数据的不完整计算）
    df['rsi'] = compute_rsi(df['close'].copy(), period=14)  # 确保传入副本避免原地修改

    # 布林带计算（同步修正窗口参数）
    df['bollinger_upper'], df['bollinger_lower'] = compute_bollinger_bands(
        df['close'].copy(),
        window=20 # 添加到函数参数中
    )

    # 动量计算（使用过去5天的变化）
    df['momentum'] = df['close'].pct_change(periods=5)

    # 波动率计算（修正窗口参数）
    df['volatility'] = df['close'].rolling(window=20, min_periods=20).std()

    # ========== 关键修复3：分阶段删除NaN ==========
    # 第一阶段：删除技术指标产生的NaN
    tech_cols = ['ma_20', 'ma_50', 'rsi', 'bollinger_upper', 'bollinger_lower', 'volatility']
    df.dropna(subset=tech_cols, how='any', inplace=True)

    # 计算滞后收益率（必须在技术指标处理后）
    for lag in [1, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    # 第二阶段：删除滞后收益率产生的NaN
    df.dropna(subset=[f'returns_lag_{lag}' for lag in [1, 3, 5]], inplace=True)

    # ========== 目标变量处理 ==========
    PREDICTION_WINDOW = 5
    df['target'] = calculate_target(df['close'], window=PREDICTION_WINDOW)
    df.dropna(subset=['target'], inplace=True)

    # 第三阶段：删除目标变量的NaN（最后PREDICTION_WINDOW行）
    df.dropna(subset=['target'], inplace=True)

    # ========== 实时数据更新特殊处理 ==========
    if len(df) > 0:
        # 对最新数据点进行插值处理（防止实时数据缺失）
        last_row = df.iloc[-1:]
        if last_row.isna().any().any():
            # 前向填充最新数据点的NaN（仅限部分特征）
            fill_cols = ['returns_lag_1', 'returns_lag_3', 'returns_lag_5']
            df[fill_cols] = df[fill_cols].ffill()
            print("警告：最新数据存在NaN，已执行前向填充")

    # ========== 最终验证 ==========
    required_columns = ['bollinger_upper', 'bollinger_lower', 'momentum',
                        'returns_lag_1', 'returns_lag_3', 'returns_lag_5', 'target']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失特征列: {missing_columns}")

    # 验证最新数据完整性
    if not df.iloc[-1:].isna().any().any():
        print(f"最新数据时间: {df.index[-1]} | 收盘价: {df['close'].iloc[-1]:.2f}")
        print(f"特征示例: MA20={df['ma_20'].iloc[-1]:.2f}, RSI={df['rsi'].iloc[-1]:.2f}")
    else:
        raise ValueError("最新数据存在无效值")
    # print(df)
    return df

# 更新实时数据（保留合理历史长度）
def update_historical_data():
    global historical_data1

    response = market_data_api.get_candlesticks(
        instId=symbol,
        bar=timeframe,
        limit=300
    )
    all_data=[]
    if response['code'] == '0':
        all_data.extend(response['data'])
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'volCcy', 'volCcyQuote', 'confirm'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        df.set_index('timestamp', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[
            ['open', 'high', 'low', 'close', 'volume']
        ].astype(float)

        # 处理特征（允许部分NaN）

        historical_data1 = process_features(df)
        # print(historical_data1)





        # print(new_df)


        # 合并数据（保留最多5000条）

        # print(historical_data)
        print(f"数据更新完成 | 当前数据量: {len(historical_data1)} 条")


# 主程序流程
if __name__ == "__main__":
    # 尝试加载现有模型
    if not load_model():
        # 需要重新训练
        historical_data=initialize_historical_data()
        train_and_save_model(historical_data)

    # 验证模型有效性
    if model is None:
        raise Exception("无有效模型可用")

    # 进入交易循环
    print("\n----- 进入交易监控模式 -----")
    while True:
        # try:
            # 每分钟更新时间戳对齐


            # 更新数据（保持数据新鲜度）
              # 需要保留原有更新逻辑


            # 执行交易
            update_historical_data()
            execute_trade()


            now = datetime.now()
            sleep_time = 600 - (now.second + now.microsecond / 1e6)
            time.sleep(max(sleep_time, 0))


        # except:
        #     print('执行错误')
        #     pass


