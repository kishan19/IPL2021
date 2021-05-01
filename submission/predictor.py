### Custom definitions and classes if any ###

import pandas as pd
import numpy as np
import joblib

##Load data file
vdf_lastn=pd.read_csv('vdf.csv')
tdf_lastn=pd.read_csv('tdf.csv')

##Load joblib file
venue_encoder=joblib.load('venue_encoder.joblib')
team_encoder=joblib.load('team_encoder.joblib')
minmaxscaler=joblib.load('min_max_scaler.joblib')
model=joblib.load('model.joblib')

def predictRuns(testInput):

    ### Your Code Here ###

    test_df=pd.read_csv(testInput)
    tdf=test_df.loc[:,['venue','innings','batting_team']]
    tdf=tdf.merge(vdf_lastn,on=['venue','innings'],suffixes=('','_venueavg'))
    tdf=tdf.merge(tdf_lastn,on=['batting_team','innings'],suffixes=('','_teamavg'))

    tdf['venue_encoded']=venue_encoder.transform(tdf['venue'])
    tdf['team_encoded']=team_encoder.transform(tdf['batting_team'])

    ##Change columns for DF
    tdf.columns=['venue', 'innings', 'batting_team', 'total_runs_venueavg', 'wicket_type',
       'total_runs_teamavg', 'wicket_type_teamavg', 'venue_encoded',
       'team_encoded']

    tdf_x=tdf.loc[:,['innings','total_runs_venueavg','total_runs_teamavg','venue_encoded','team_encoded','wicket_type','wicket_type_teamavg']]

    Xt=minmaxscaler.transform(tdf_x)
    prediction=int(np.round(model.predict(Xt),0))

    ##Check wickets lost in the Powerplay
    try:
      wkt_factor=(test_df['batsmen'].apply(lambda x:len(x.split(',')))[0]-2)*7
    except Exception as ex:
       wkt_factor=0
       pass

    ##Adjust prediction for wickets lost in PP
    #prediction=prediction-wkt_factor

    return prediction
