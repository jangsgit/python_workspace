from flask import Flask, render_template, request
import plotly.express as px
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    # 사용자로부터 입력 받기
    data = request.form['data']
    data = [int(x) for x in data.split(',')]
    
    # 데이터 프레임 생성
    df = pd.DataFrame({'Value': data})
    
    # Plotly를 사용하여 선 그래프 생성
    fig = px.line(df, y='Value', title='Sample Plot')

    # 그래프를 HTML로 변환하여 전달
    graph_html = fig.to_html(full_html=False)

    return render_template('plot.html', graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)