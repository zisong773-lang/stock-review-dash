import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import os
import json
import textwrap
import base64
import io
import toml
from datetime import datetime, timedelta
import numpy as np

# --- 基础库与配置检查 ---
try:
    import yfinance as yf
    import s3fs
except ImportError as e:
    print(f"缺少必要库: {e}")

# --- 尝试读取 Secrets ---
SECRETS = {}
try:
    if os.path.exists(".streamlit/secrets.toml"):
        SECRETS = toml.load(".streamlit/secrets.toml")
    elif "aws" in os.environ:
        pass
except Exception as e:
    print(f"读取配置失败: {e}")

# --- 云端连接初始化 ---
USE_CLOUD = False
fs = None
BUCKET_NAME = ""
HISTORY_DIR = ""

if "aws" in SECRETS:
    try:
        fs = s3fs.S3FileSystem(
            key=SECRETS["aws"]["aws_access_key_id"],
            secret=SECRETS["aws"]["aws_secret_access_key"]
        )
        BUCKET_NAME = SECRETS["aws"]["bucket_name"]
        HISTORY_DIR = f"{BUCKET_NAME}/history_charts"
        if not fs.exists(HISTORY_DIR):
            fs.makedirs(HISTORY_DIR)
        USE_CLOUD = True
        print("✅ AWS S3 连接成功")
    except Exception as e:
        print(f"AWS S3 连接失败: {e}")
else:
    print("⚠️ 未找到 AWS 配置，云端功能禁用")


# --- 初始化 Dash 应用 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="股价复盘系统 (Dash版)")
server = app.server

# --- 辅助函数 ---

def process_text_smart(text, wrap_width):
    if not isinstance(text, str): return str(text)
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        line = line.replace("<br>", "\n")
        sub_lines = line.split("\n")
        for sl in sub_lines:
            wrapped = textwrap.wrap(sl, width=wrap_width)
            processed_lines.extend(wrapped)
    return "<br>".join(processed_lines)

# 百分比格式化函数
def format_pct(value):
    """
    将数值转换为保留1位小数的百分比字符串。
    例如: 0.0312 -> 3.1%
    """
    if pd.isna(value) or value == '':
        return ""
    try:
        # 尝试转换为浮点数
        f_val = float(value)
        # 乘以100并保留1位小数
        return f"{f_val * 100:.1f}%"
    except (ValueError, TypeError):
        # 如果转换失败（例如已经是字符串），则原样返回
        return str(value)

def generate_mock_data(start, end):
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)
    if n == 0: return None
    np.random.seed(42)
    returns = np.random.normal(loc=0.0003, scale=0.015, size=n)
    price = 3000 * np.cumprod(1 + returns)
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = df['Close'].shift(1).fillna(price[0]) * (1 + np.random.randn(n)*0.005)
    return df.round(0)

def find_col_in_list(columns, keywords, exclude_keywords=None):
    for col in columns:
        col_str = str(col)
        if exclude_keywords and any(ex in col_str for ex in exclude_keywords):
            continue
        for kw in keywords:
            if kw in col_str:
                return col
    return None

def extract_table_dynamically(df, required_keywords, name="Table"):
    def check_columns(cols):
        found_cols = {}
        for key, (kws, ex_kws) in required_keywords.items():
            found = find_col_in_list(cols, kws, ex_kws)
            if found:
                found_cols[key] = found
            else:
                return None
        return found_cols

    found_cols = check_columns(df.columns)
    if found_cols:
        return df, found_cols

    max_scan = min(len(df), 100)
    for i in range(max_scan):
        row_values = df.iloc[i].astype(str).tolist()
        is_header_row = True
        for key, (kws, ex_kws) in required_keywords.items():
            if not any(kw in cell for cell in row_values for kw in kws):
                is_header_row = False
                break
        
        if is_header_row:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i]
            new_found_cols = check_columns(new_df.columns)
            if new_found_cols:
                return new_df, new_found_cols
    return None, None

def aggregate_details(df, group_keys, detail_col, output_detail_name="Detail"):
    if not detail_col: return df
    for k in group_keys:
        df[k] = df[k].ffill()
    
    def join_text(series):
        texts = [str(s).strip() for s in series if pd.notna(s) and str(s).strip() != '']
        if not texts: return None
        if len(texts) == 1: return texts[0]
        return "<br>".join([f"• {t}" for t in texts])

    agg_dict = {detail_col: join_text}
    temp = df.groupby(group_keys, as_index=False).agg(agg_dict)
    temp = temp.rename(columns={detail_col: output_detail_name})
    return temp

def parse_excel_content(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        file = io.BytesIO(decoded)
        all_sheets = pd.read_excel(file, sheet_name=None)
        events_list = []
        phases_list = []
        prices_df = None
        
        if 'Prices' in all_sheets:
            prices_df = all_sheets['Prices']
            prices_df['Date'] = pd.to_datetime(prices_df['Date'])
            prices_df.set_index('Date', inplace=True)

        event_rules = {
            'event': (['主要驱动', 'Event'], None),
            'date': (['日期', 'Date', '时间'], ['起始', '开始', 'Start', '结束', 'End'])
        }
        
        phase_rules = {
            'phase': (['阶段概述', 'Phase'], None),
            'start': (['起始日期', '开始日期', 'Start'], None),
            'end': (['结束日期', 'End'], None)
        }

        for sheet_name, df in all_sheets.items():
            df.columns = df.columns.astype(str).str.strip()
            
            # --- 1. 提取事件表 (Events) ---
            e_df, e_cols = extract_table_dynamically(df, event_rules, "Events")
            if e_df is not None:
                hover_col = find_col_in_list(e_df.columns, ['详细解释', '因果链', 'Detailed'])
                change_col = find_col_in_list(e_df.columns, ['涨跌幅', '幅度', 'Change', 'Pct', '%'])

                cols_to_keep = [e_cols['date'], e_cols['event']]
                if hover_col: cols_to_keep.append(hover_col)
                if change_col: cols_to_keep.append(change_col)
                
                temp = e_df[cols_to_keep].copy()
                
                group_cols = [e_cols['date'], e_cols['event']]
                if change_col: group_cols.append(change_col)

                if hover_col:
                    temp = aggregate_details(temp, group_keys=group_cols, detail_col=hover_col, output_detail_name='详细解释')
                
                rename_dict = {e_cols['date']: 'Date', e_cols['event']: '主要驱动'}
                if change_col: rename_dict[change_col] = '日涨跌幅'
                
                temp = temp.rename(columns=rename_dict)
                temp['Date'] = pd.to_datetime(temp['Date'], errors='coerce')
                temp = temp.dropna(subset=['Date'])
            
                if not temp.empty:
                    events_list.append(temp)
            
            # --- 2. 提取阶段表 (Phases) ---
            p_df, p_cols = extract_table_dynamically(df, phase_rules, "Phases")
            if p_df is not None:
                hover_col = find_col_in_list(p_df.columns, ['关键因素', '要点', 'Key Factors'])
                range_col = find_col_in_list(p_df.columns, ['区间涨跌幅', '区间', 'Range', 'Return'])

                cols_to_keep = [p_cols['start'], p_cols['end'], p_cols['phase']]
                if hover_col: cols_to_keep.append(hover_col)
                if range_col: cols_to_keep.append(range_col) 

                temp = p_df[cols_to_keep].copy()
                
                group_cols = [p_cols['start'], p_cols['end'], p_cols['phase']]
                if range_col: group_cols.append(range_col)

                if hover_col:
                    temp = aggregate_details(temp, group_keys=group_cols, detail_col=hover_col, output_detail_name='关键因素')
                
                rename_dict = {p_cols['start']: 'Start date', p_cols['end']: 'End date', p_cols['phase']: '阶段概述'}
                if range_col: rename_dict[range_col] = '区间涨跌幅'

                temp = temp.rename(columns=rename_dict)
                temp['Start date'] = pd.to_datetime(temp['Start date'], errors='coerce')
                temp['End date'] = pd.to_datetime(temp['End date'], errors='coerce')
                temp = temp.dropna(subset=['Start date'])
                if not temp.empty:
                    phases_list.append(temp)

        events_df = pd.concat(events_list, ignore_index=True) if events_list else None
        phases_df = pd.concat(phases_list, ignore_index=True) if phases_list else None
        return events_df, phases_df, prices_df

    except Exception as e:
        print(f"解析出错: {e}")
        return None, None, None

def get_yahoo_data(ticker, start, end, proxy_url=None):
    if proxy_url:
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
    else:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        
    try:
        dat = yf.Ticker(ticker)
        df = dat.history(start=start, end=end, auto_adjust=True)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"Yahoo Err: {e}")
        return pd.DataFrame()

def apply_relayout_to_fig(fig_dict, relayout_data):
    if not relayout_data:
        return fig_dict
    if hasattr(fig_dict, 'to_dict'):
        fig_dict = fig_dict.to_dict()
    if 'layout' not in fig_dict:
        fig_dict['layout'] = {}

    for key, value in relayout_data.items():
        if 'annotations' in key:
            try:
                parts = key.split('.')
                idx = int(parts[0].replace('annotations[', '').replace(']', ''))
                attr = parts[1] 
                if 'annotations' not in fig_dict['layout']:
                    fig_dict['layout']['annotations'] = []
                if idx < len(fig_dict['layout']['annotations']):
                    fig_dict['layout']['annotations'][idx][attr] = value
            except: pass
        elif 'shapes' in key:
            try:
                parts = key.split('.')
                idx = int(parts[0].replace('shapes[', '').replace(']', ''))
                attr = parts[1]
                if 'shapes' not in fig_dict['layout']:
                    fig_dict['layout']['shapes'] = []
                if idx < len(fig_dict['layout']['shapes']):
                    fig_dict['layout']['shapes'][idx][attr] = value
            except: pass
        else:
            fig_dict['layout'][key] = value
    return fig_dict


# --- 界面布局 ---

sidebar = dbc.Card(
    [
        html.H4("🎛️ 设置", className="card-title"),
        html.Hr(),
        dbc.Label("系统模式"),
        dbc.RadioItems(
            options=[
                {"label": "🚀 生成新图表", "value": "new"},
                {"label": "📂 云端历史记录", "value": "history"},
            ],
            value="new",
            id="app-mode-selector",
            className="mb-3",
        ),
        
        # --- 【修改】将保存区域移动到侧边栏，位置醒目 ---
        html.Div([
            html.Hr(),
            dbc.Card([
                dbc.CardBody([
                    html.H5("💾 保存云端快照", className="card-title text-success", style={'fontSize': '1rem', 'fontWeight': 'bold'}),
                    html.Div("包含当前拖拽后的位置", className="text-muted small mb-2"),
                    dbc.Input(id="save-filename", placeholder="输入文件名 (如: TSLA_复盘)", size="sm", className="mb-2"),
                    dbc.Button("☁️ 立即保存布局", id="save-cloud-btn", color="success", size="sm", className="w-100"),
                ], className="p-2")
            ], className="mb-3 border-success", outline=True)
        ], id="save-area"),
        # -------------------------------------------
        
        html.Hr(),
        
        html.Div([
            dbc.Label("0. 代理设置"),
            dbc.Checkbox(label="开启代理", value=False, id="enable-proxy"),
            dbc.Input(id="proxy-addr", value="http://127.0.0.1:17890", type="text", className="mb-3"),
            dbc.Label("1. 数据来源"),
            dbc.RadioItems(
                options=[
                    {"label": "Yahoo Finance", "value": "yahoo"},
                    {"label": "Excel Prices表", "value": "excel_price"},
                    {"label": "模拟数据", "value": "mock"},
                ],
                value="yahoo",
                id="data-source-select",
                className="mb-3",
            ),
            dbc.Label("2. 时间与代码"),
            dbc.Input(id="ticker-input", value="6324.T", type="text", placeholder="股票代码", className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Input(id="start-date", value="2024-12-23", type="date")),
                dbc.Col(dbc.Input(id="end-date", value=datetime.today().strftime("%Y-%m-%d"), type="date")),
            ], className="mb-3"),
            dbc.Label("3. 上传 Excel (含事件/阶段)"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['拖拽或点击上传']),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px 0'
                },
                multiple=False
            ),
            html.Div(id='output-file-name', className="text-muted small mb-3"),
            html.Hr(),
            dbc.Label("4. 视觉微调"),
            dbc.Label("导出倍率", html_for="export-scale"),
            dbc.RadioItems(
                options=[{"label": "1x", "value": 1}, {"label": "2x", "value": 2}, {"label": "3x", "value": 3}],
                value=1, id="export-scale", inline=True, className="mb-2"
            ),
            dbc.Label("字体大小 (阶段 / 事件)"),
            dcc.Slider(id="phase-font-size", min=10, max=80, value=20, marks=None, tooltip={"placement": "bottom"}),
            dcc.Slider(id="event-font-size", min=8, max=60, value=16, marks=None, tooltip={"placement": "bottom"}),
            dbc.Label("布局间距 (阶段高度 / 底部留白)"),
            dcc.Slider(id="phase-label-y", min=1.0, max=1.3, step=0.01, value=1.02, marks=None),
            dcc.Slider(id="bottom-margin", min=50, max=200, value=80, marks=None),
            dbc.Label("标签换行 (阶段 / 事件)"),
            dcc.Slider(id="label-wrap-width", min=5, max=50, value=10, marks=None),
            dbc.Label("悬浮提示换行字数"),
            dcc.Slider(id="hover-wrap-width", min=20, max=80, value=40, marks=None),
            dbc.Label("防重叠 (引线长度 / 阶梯)"),
            dcc.Slider(id="arrow-len-base", min=20, max=150, value=50, marks=None),
            dcc.Slider(id="stagger-steps", min=3, max=15, value=6, marks=None),
            html.Br(),
            dbc.Button("🔄 更新图表", id="update-btn", color="primary", className="w-100 mb-3"),
        ], id="control-panel-new"),
        
        html.Div([
            dbc.Button("🔄 刷新列表", id="refresh-list-btn", color="secondary", size="sm", className="mb-3"),
            dbc.Label("搜索文件"),
            dbc.Input(id="search-history", placeholder="输入文件名过滤...", className="mb-2"),
            dbc.Label("选择文件"),
            dcc.Dropdown(id="history-file-dropdown", options=[], placeholder="选择图表..."),
            html.Br(),
            dbc.Button("🗑️ 删除选中文件", id="delete-btn", color="danger", outline=True, size="sm", className="w-100"),
        ], id="control-panel-history", style={'display': 'none'}),
    ],
    body=True,
    style={"height": "100vh", "overflow-y": "scroll"}
)

content = html.Div(
    [
        html.H2("📈 2025 股价复盘系统 (Dash Cloud)", className="display-6"),
        html.Hr(),
        html.Div(id="msg-area"),
        dcc.Loading(
            dcc.Graph(
                id='main-graph', 
                style={'height': '85vh'}, 
                config={'editable': True, 'scrollZoom': True, 'displayModeBar': True, 'showLink': False}
            )
        ),
        # --- 【修改】此处原本的 save-area 已移除 ---
    ],
    className="p-4"
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className="bg-light"),
                dbc.Col(content, width=9),
            ],
            className="g-0",
        ),
        dcc.Store(id='store-excel-data'), 
    ],
    fluid=True,
)


# --- Callbacks ---

@app.callback(
    [Output("control-panel-new", "style"),
     Output("control-panel-history", "style"),
     Output("save-area", "style")],
    [Input("app-mode-selector", "value")]
)
def toggle_mode(mode):
    if mode == "new":
        return {'display': 'block'}, {'display': 'none'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}

@app.callback(
    [Output('store-excel-data', 'data'),
     Output('output-file-name', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def parse_file(contents, filename):
    if contents is None:
        return None, ""
    
    events, phases, prices = parse_excel_content(contents, filename)
    
    data = {
        'events': events.to_json(date_format='iso', orient='split') if events is not None else None,
        'phases': phases.to_json(date_format='iso', orient='split') if phases is not None else None,
        'prices': prices.to_json(date_format='iso', orient='split') if prices is not None else None
    }
    return data, f"已加载: {filename}"

@app.callback(
    Output('main-graph', 'figure'),
    [Input('update-btn', 'n_clicks'),
     Input('history-file-dropdown', 'value')], 
    [State('app-mode-selector', 'value'),
     State('data-source-select', 'value'),
     State('ticker-input', 'value'),
     State('start-date', 'value'),
     State('end-date', 'value'),
     State('enable-proxy', 'value'),
     State('proxy-addr', 'value'),
     State('store-excel-data', 'data'),
     State('phase-font-size', 'value'),
     State('event-font-size', 'value'),
     State('phase-label-y', 'value'),
     State('bottom-margin', 'value'),
     State('label-wrap-width', 'value'),
     State('hover-wrap-width', 'value'),
     State('arrow-len-base', 'value'),
     State('stagger-steps', 'value'),
     State('export-scale', 'value')] 
)
def update_chart(n_updates, history_file, mode, 
                 source, ticker, start, end, use_proxy, proxy_addr, 
                 excel_data, 
                 p_fs, e_fs, p_y, b_margin, wrap_w, 
                 hover_w,
                 arrow_len, stag_steps, scale):
    
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if mode == "history":
        if not history_file or not USE_CLOUD:
            return go.Figure()
        try:
            full_path = history_file
            with fs.open(full_path, 'r') as f:
                fig_json = json.load(f)
            fig = go.Figure(fig_json)
            fig.update_layout(dragmode='pan')
            return fig
        except Exception as e:
            print(f"Load error: {e}")
            return go.Figure()

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + timedelta(days=1)
    
    stock_df = None
    events_df, phases_df = None, None
    
    if excel_data:
        if excel_data.get('events'):
            events_df = pd.read_json(io.StringIO(excel_data['events']), orient='split')
            events_df['Date'] = pd.to_datetime(events_df['Date'])
        if excel_data.get('phases'):
            phases_df = pd.read_json(io.StringIO(excel_data['phases']), orient='split')
            phases_df['Start date'] = pd.to_datetime(phases_df['Start date'])
            phases_df['End date'] = pd.to_datetime(phases_df['End date'])
            
    if source == 'yahoo':
        stock_df = get_yahoo_data(ticker, start, end_dt.strftime('%Y-%m-%d'), proxy_addr if use_proxy else None)
    elif source == 'excel_price':
        if excel_data and excel_data.get('prices'):
            stock_df = pd.read_json(io.StringIO(excel_data['prices']), orient='split')
            stock_df = stock_df[(stock_df.index >= start_dt) & (stock_df.index <= end_dt)]
    else:
        stock_df = generate_mock_data(start_dt, end_dt)

    if stock_df is None or stock_df.empty:
        fig = go.Figure()
        fig.update_layout(title="暂无数据")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', 
                             name=f"{ticker} Close", line=dict(color='#1976D2', width=2.5)))
    
    data_start, data_end = stock_df.index.min(), stock_df.index.max()
    y_min, y_max = stock_df['Close'].min(), stock_df['Close'].max()

    # --- 绘制阶段 (Phases) ---
    if phases_df is not None and not phases_df.empty:
        phase_colors = ["rgba(255,99,132,0.12)", "rgba(54,162,235,0.12)", "rgba(255,206,86,0.15)", "rgba(75,192,192,0.12)"]
        target_col = find_col_in_list(phases_df.columns, ['阶段概述'])
        
        for i, row in phases_df.iterrows():
            p_s = max(row['Start date'], data_start)
            p_e = min(row['End date'], data_end)
            if p_s < p_e:
                mid = p_s + (p_e - p_s) / 2
                fig.add_vrect(x0=p_s, x1=p_e, fillcolor=phase_colors[i % 4], layer="below", line_width=0)
                
                # --- 文本构建 ---
                main_txt = str(row.get(target_col, ''))
                wrapped_main = process_text_smart(main_txt, wrap_w)
                
                # 获取区间涨跌幅并格式化
                range_chg = ""
                if '区间涨跌幅' in row and pd.notna(row['区间涨跌幅']):
                    range_val = format_pct(row['区间涨跌幅'])
                    if range_val:
                        range_chg = f"<br><span style='font-size:0.8em'>({range_val})</span>"
                
                display_html = f"<b>{wrapped_main}</b>{range_chg}"

                # 悬浮文本
                hover_txt = ""
                if '关键因素' in row:
                    hover_txt = process_text_smart(str(row['关键因素']), hover_w)
                else:
                    hover_txt = process_text_smart(main_txt, hover_w)
                
                cy = p_y + (0.05 if (i % 2) != 0 else 0)
                fig.add_annotation(
                    x=mid, y=cy, yref="paper", 
                    text=display_html, 
                    showarrow=False,
                    font=dict(size=p_fs, color="#555"),
                    bgcolor="rgba(255,255,255,0.8)", borderpad=3,
                    hovertext=hover_txt,
                    captureevents=True
                )

    # --- 绘制事件 (Events) ---
    if events_df is not None and not events_df.empty:
        events_df = events_df.sort_values('Date')
        label_col = find_col_in_list(events_df.columns, ['主要驱动'])
        
        for i, row in events_df.iterrows():
            edate = row['Date']
            if data_start <= edate <= data_end:
                try:
                    idx = stock_df.index.get_indexer([edate], method='nearest')[0]
                    curr_date = stock_df.index[idx]
                    price = stock_df.loc[curr_date]['Close']
                    if isinstance(price, pd.Series): price = price.iloc[0]
                    
                    prev_price = stock_df['Close'].iloc[idx-1] if idx > 0 else price
                    is_rising = price >= prev_price
                    color = "#D32F2F" if is_rising else "#00796B"
                    ay_dir = 1 if is_rising else -1
                    
                    stagger = i % stag_steps
                    a_len = arrow_len + (stagger * 50)
                    
                    # --- 文本构建 ---
                    # 1. 日期
                    date_str = edate.strftime('%m-%d')
                    
                    # 2. 涨跌幅
                    change_str = ""
                    if '日涨跌幅' in row and pd.notna(row['日涨跌幅']):
                        val = format_pct(row['日涨跌幅'])
                        if val: change_str = f" {val}"
                    
                    # 3. 事件内容
                    event_txt = str(row.get(label_col, ''))
                    wrapped_event = process_text_smart(event_txt, wrap_w)
                    
                    # 拼装: [日期 涨幅] <换行> [事件]
                    display_html = f"<b>{date_str}{change_str}</b><br>{wrapped_event}"
                    
                    # 悬浮文本
                    hover_txt = ""
                    if '详细解释' in row:
                        hover_txt = process_text_smart(str(row['详细解释']), hover_w)
                    else:
                        hover_txt = process_text_smart(event_txt, hover_w)
                    
                    fig.add_annotation(
                        x=curr_date, y=price,
                        text=display_html,
                        showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=color,
                        ax=0, ay=a_len * ay_dir,
                        font=dict(size=e_fs, color="#333"),
                        bgcolor="rgba(255,255,255,0.8)", bordercolor=color,
                        hovertext=hover_txt,
                        hoverlabel=dict(bgcolor="white", font=dict(size=e_fs)),
                        captureevents=True
                    )
                except: pass

    fig.update_layout(
        title=dict(text=f"{ticker} 复盘 (Dash版)", x=0.5),
        yaxis_title="Price",
        height=900,
        margin=dict(t=150, b=b_margin),
        template="plotly_white",
        hovermode="x unified",
        dragmode="pan"
    )
    return fig

@app.callback(
    Output("msg-area", "children"),
    Input("save-cloud-btn", "n_clicks"),
    [State("save-filename", "value"),
     State("ticker-input", "value"),
     State("main-graph", "figure"),       
     State("main-graph", "relayoutData")] 
)
def save_chart_to_cloud(n, filename, ticker, fig_data, relayout_data):
    if not n: return ""
    if not USE_CLOUD:
        return dbc.Alert("❌ 未配置 AWS S3", color="danger")
    
    try:
        final_fig_dict = apply_relayout_to_fig(fig_data, relayout_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c for c in filename if c.isalnum() or c in (' ', '_', '-')]).strip() if filename else "Untitled"
        s3_name = f"{timestamp}_{ticker}_{safe_name}.json"
        path = f"{HISTORY_DIR}/{s3_name}"
        
        with fs.open(path, "w") as f:
            json.dump(final_fig_dict, f)
            
        return dbc.Alert(f"✅ 保存成功 (含拖拽): {s3_name}", color="success", dismissable=True)
    except Exception as e:
        return dbc.Alert(f"保存失败: {e}", color="danger", dismissable=True)

@app.callback(
    Output("history-file-dropdown", "options"),
    [Input("control-panel-history", "style"), 
     Input("refresh-list-btn", "n_clicks"),
     Input("delete-btn", "n_clicks")], 
    State("search-history", "value")
)
def update_file_list(panel_style, n_refresh, n_del, search_term):
    if not USE_CLOUD or panel_style.get('display') == 'none':
        return no_update
    
    try:
        files = fs.glob(f"{HISTORY_DIR}/*.json")
        files_info = []
        for f in files:
            info = fs.info(f)
            files_info.append({'path': f, 'time': info['LastModified']})
        
        files_info.sort(key=lambda x: x['time'], reverse=True)
        
        options = []
        for item in files_info:
            name = os.path.basename(item['path'])
            if search_term and search_term.lower() not in name.lower():
                continue
            options.append({'label': f"{item['time'].strftime('%m-%d %H:%M')} | {name}", 'value': item['path']})
            
        return options
    except Exception as e:
        print(f"List error: {e}")
        return []

@app.callback(
    Output("delete-btn", "disabled"), 
    Input("delete-btn", "n_clicks"),
    State("history-file-dropdown", "value")
)
def delete_file(n, file_path):
    if n and file_path and USE_CLOUD:
        try:
            fs.rm(file_path)
        except: pass
    return False

if __name__ == "__main__":
    app.run(debug=True, port=8050)
