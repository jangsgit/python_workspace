from flask import Flask, jsonify, request, render_template,  redirect, url_for, render_template_string

import pyodbc
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd 
import plotly.express as px
import json 

from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

#Anaconda install -> tensorflow install 
# import tensorflow as tf

app = Flask(__name__)

# 데이터베이스 연결 설정
server = 'emmsg.co.kr,33000'  # 서버 주소
database = 'elv_lrt'  # 데이터베이스 이름  elv_cmstest   erp_actas
username = 'actaselv'  # 사용자 이름
password = 'elv@5020'  # 비밀번호
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)

@app.route('/')
def index():
    return render_template('index.html')

###### SELECT 데이타 가져오기 
@app.route('/data', methods=["POST"])
def get_data():
    data = request.form['date_query']  
    cursor = cnxn.cursor()
    cursor.execute("SELECT top 2 misdate, misnum, cltcd FROM tb_da023 where misdate = '" + data + "' ")  # 쿼리 예시
    rows = cursor.fetchall()
    cursor.close() 
     
    # 결과를 JSON으로 변환
    results = []
    for row in rows:
        results.append({ 'misdate': row.misdate, 'misnum': row.misnum, 'cltcd': row.cltcd })  # 컬럼 이름을 적절하게 변경하세요.
        answer =  jsonify(results)
        
        # return redirect(url_for('show_answer', answer=answer)) 
        return jsonify(results)



###### SELECT 데이타 그래프 표현 
@app.route('/graph', methods=["POST"])
def graph_data():
    data = request.form['year_query']  
    cursor = cnxn.cursor()
    cursor.execute("SELECT substring(misdate,3,4) as yymm, Round(sum(misamt) * 0.01 , 0) as misamt  FROM tb_da023 where left(misdate,4) = '" + data + "'  group by substring(misdate,3,4) order by  substring(misdate,3,4)  ")  # 쿼리 예시    
    rows = cursor.fetchall()
    cursor.close() 
    
    # 그래프 생성
    x_values = [row[0] for row in rows]
    y_values = [row[1] for row in rows] 
    plt.figure()
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False 
    plt.plot(x_values, y_values)
    plt.title(data + '년 매출현황')
    plt.xlabel('년월')
    plt.ylabel('매출액')

    # 그래프를 base64 인코딩된 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()  
    return render_template_string('''<img src="data:image/png;base64,{{graph_url}}" alt="Graph">''', graph_url=graph_url)




###### 기계학습모델
@app.route('/accline', methods=["POST"])
def acc_data():
    acc_query = request.form['acc_query']  
    cursor = cnxn.cursor()
    ls_sql = "select   A.print_content AS Description, B.accnm as AccountTitle from tb_bank_accsave  A WITH (NOLOCK) ,  TB_AA010 B WITH (NOLOCK) "
    ls_sql  = ls_sql + " WHERE A.acc_spdate + A.acc_spnum = B.spdate + B.spnum and len(A.print_content) > 0  and b.acccd <> '1015'  and left(spdate,4) in ('2022', '2023' ) "    
    cursor.execute(ls_sql)  # 쿼리 예시
    rows  = cursor.fetchall()
    cursor.close() 
    # print(rows)
    # 데이터를 pandas DataFrame으로 변환
    data = pd.DataFrame([tuple(row) for row in rows], columns=['Description', 'AccountTitle']) 
    
    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(data['Description'], data['AccountTitle'], test_size=0.2, random_state=42)
 
    # 텍스트를 TF-IDF 벡터로 변환하고 나이브 베이즈 분류기 학습
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train) 
    # 모델 성능 평가
    print("Test Score: ", model.score(X_test, y_test))
    print("Test Sacc_query: ", acc_query)
    predicted_account_title = model.predict([acc_query])[0]
    print("Predicted Account Title: ", predicted_account_title)

    return "Predicted Account Title: "  + predicted_account_title



###### 랜덤 포레스트     pip install pyodbc scikit-learn pandas tensorflow
@app.route('/accrandom', methods=["POST"])
def accrandom_data():
    acc_query = request.form['random_query']  
    cursor = cnxn.cursor()
    ls_sql = "select   A.print_content AS Description, B.accnm as AccountTitle from tb_bank_accsave  A WITH (NOLOCK) ,  TB_AA010 B WITH (NOLOCK) "
    ls_sql  = ls_sql + " WHERE A.acc_spdate + A.acc_spnum = B.spdate + B.spnum and len(A.print_content) > 0  and b.acccd <> '1015'  and left(spdate,4) in ('2022', '2023' ) "    
    df = pd.read_sql(ls_sql,cnxn) 
    #데이타 전처리  
    vectorizer = TfidfVectorizer(max_features=1000)

    #특성과 타겟 분리  
    X = vectorizer.fit_transform(df['Description'])
    y = df['AccountTitle']

    #데이타 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #랜덤 포레스트 모델 학습
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    #예측 및 평가  
    rf_predictions = rf_model.predict(X_test)
    # 모델 성능 평가
    print("Test Score: ", accuracy_score(y_test, rf_predictions))
    print("Test Sacc_query: ", acc_query)
    acc_query_vect = vectorizer.transform([acc_query]) 
    print("Predicted Account Title: ", rf_model.predict(acc_query_vect))
    predicted_account_title = rf_model.predict(acc_query_vect)  
    return "Predicted Account Title: "  + str(predicted_account_title)


###### 그래디언트 부스팅
@app.route('/accgrad', methods=["POST"])
def accgrad_data():
    acc_query = request.form['grad_query']  
    cursor = cnxn.cursor()
    ls_sql = "select   A.print_content AS Description, B.accnm as AccountTitle from tb_bank_accsave  A WITH (NOLOCK) ,  TB_AA010 B WITH (NOLOCK) "
    ls_sql  = ls_sql + " WHERE A.acc_spdate + A.acc_spnum = B.spdate + B.spnum and len(A.print_content) > 0  and b.acccd <> '1015'  and left(spdate,4) in ('2022', '2023' ) "    
    df = pd.read_sql(ls_sql,cnxn) 
    #데이타 전처리  
    vectorizer = TfidfVectorizer(max_features=1000)

    #특성과 타겟 분리  
    X = vectorizer.fit_transform(df['Description'])
    y = df['AccountTitle']

    #데이타 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #랜덤 포레스트 모델 학습
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    #예측 및 평가  
    gb_predictions = gb_model.predict(X_test)
    # 모델 성능 평가
    print("Test Score: ", accuracy_score(y_test, gb_predictions))
    print("Test Sacc_query: ", acc_query)
    acc_query_vect = vectorizer.transform([acc_query]) 
    print("Predicted Account Title: ", gb_model.predict(acc_query_vect))
    predicted_account_title = gb_model.predict(acc_query_vect)  
    return "Predicted Account Title: "  + str(predicted_account_title)



###### 랜덤 포레스트 고장원인예측
@app.route('/elvrandom', methods=["POST"])
def accelv_data():
    ho_query = request.form['ho_query'] 
    year_query = request.form['year_query'] 
    # falut_model(acc_query) 
    # acc_query = arg
    cursor = cnxn.cursor()
    ls_sql = " select equpnm as ApartUnit, b.contnm as FaultReason, count(a.custcd) as Count  "
    ls_sql  = ls_sql + " from  tb_e401 a, tb_e010 b "
    ls_sql  = ls_sql + " where a.contcd = b.contcd and a.cltcd = '00188' and a.actcd ='00474' and LEN(isnull(a.contcd,'')) > 0 "
    ls_sql  = ls_sql + " group by equpnm, b.contnm "
    df = pd.read_sql(ls_sql,cnxn) 
    #데이타 전처리   
    le = LabelEncoder()
    df['FaultReason'] = le.fit_transform(df['FaultReason'])
    df['ApartUnit'] = le.fit_transform(df['ApartUnit'])
    #특성과 타겟 분리  
    X = df[['ApartUnit', 'Year']]
    y = df['FaultReason']

    #데이타 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('11111111111')
    #랜덤 포레스트 모델 학습
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print('2222222222') 

    prediction = rf_model.predict([[le.transform([ho_query])[0], int(year)]])
    prediction_fault = le.inverse_transform(prediction)[0]

    #시각화 
    fig = px.bar(x=['Predicted Fault'], y=[prediction_fault], labels={'x':'Fault', 'y':'Reason'})
    graphJSON = json.dump(fig, )


    return render_template('index.html', plot_rul=plot_rul)


def falut_model(arg): 
    acc_query = arg
    cursor = cnxn.cursor()
    ls_sql = " select equpnm, b.contnm as FaultReason, count(a.custcd) as Count  "
    ls_sql  = ls_sql + " from  tb_e401 a, tb_e010 b "
    ls_sql  = ls_sql + " where a.contcd = b.contcd and a.cltcd = '00188' and a.actcd ='00474' and LEN(isnull(a.contcd,'')) > 0 "
    ls_sql  = ls_sql + " group by equpnm, b.contnm "
    df = pd.read_sql(ls_sql,cnxn) 
    #데이타 전처리   
    le = LabelEncoder()
    df['FaultReason_encoded'] = le.fit_transform(df['FaultReason'])
    #특성과 타겟 분리  
    X = df[['equpnm', 'FaultReason_encoded']]
    y = df['Count']

    #데이타 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('11111111111')
    #랜덤 포레스트 모델 학습
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print('2222222222')
    #예측 및 평가  
    rf_predictions = rf_model.predict(X_test)
    print('333333333')
    #결과 시각화 
    plt.scatter(X_test['equpnm'], rf_predictions) 
    plt.xlabel('equpnm')
    plt.ylabel('FaultReason')
    plt.title('풍무자이1단지고장예측')
    plt.show()


###### CHAT GPT 질의하기  
@app.route('/query-gpt', methods=["POST"])
def query_gpt4():
    #  data = request.json
     data = request.form['user_query'] 
     prompt = data  #data.get('prompt')
     client = OpenAI(api_key="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")

     if not prompt :
          return jsonify({'error': 'No prompt provided'}), 400 
     try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={ "type": "json_object" },
            messages=[
                 {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                 {"role": "user", "content": prompt}
            ],
        )
        answer =  response.choices[0].message.content
        return redirect(url_for('show_answer', answer=answer))   #jsonify({'response' : response.choices[0].message.content})
     except Exception as e:
         return jsonify({'error': str(e)}), 500


@app.route('/show-answer')
def show_answer():
    answer = request.args.get('answer')
    print(answer)
    return f"<h1>Answer from GPT : </h1><p>{answer}</p>"

###### MAIN 
if __name__ == '__main__':
#   app.run(debug=True) #port=1000   #요거는 내부테스트용
    app.run(host='0.0.0.0', port=5000)

# prompt="python flask를 이용하여 chatgpt4에 질의를 주고 받는 웹페이지 샘플은"
# response = query_gpt4(prompt)
# print(response)