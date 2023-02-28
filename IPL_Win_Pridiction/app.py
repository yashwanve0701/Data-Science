import streamlit as st
import pickle
import pandas as pd
teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']
citys=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'nan', 'Mohali', 'Bengaluru']
pipe=pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictor')
col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox('Select The Batting Team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select The Bowling Team',sorted(teams))

selected_city=st.selectbox('Select Host City',sorted(citys))

target=st.number_input('TARGET')

col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('SCORE')
with col4:
    overs=st.number_input('OVER COMPLETED')
with col5:
    wickets=st.number_input('WICKETS')

if st.button('PREDICT PROBABILITY'):
    run_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs
    rrr=(run_left*6)/balls_left

    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                           'city':[selected_city],'run_left':[run_left],'balls_left':[balls_left],
                           'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    st.table(input_df)
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.text(batting_team+"- "+str(round(win*100))+"%")
    st.text(bowling_team + "- " + str(round(loss*100)) + "%")
    st.text(result)

