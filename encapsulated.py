from functions import *
if __name__ == "__main__":
    #Main Principal, activa todas las funciones anteriores.
    #-------------tiene cosas hardcodeadas ya que este proceso sera eliminado con la incorporación del sistema------
    df=data_input()
    x,y=Trip_Clustering(df)
    #Random bound Box to see if avg is effectively calculated
    Weekly_Trips_Avg=Weekly_avg_Bounding_Box(df)
    Weekly_avg_region=Weekly_avg_region(df)
    #Only 8 trips were captured
    
    #-------Offline Training Modlu for ML
    
    
    #-------Online Training Module for ML.4
    
    #-----Output Module----------------------
    #Let's get a json with everything4
    json_output = df.to_json(orient="index")
    #Returning the processed dataframe as a SQL database
    conn = sqlite3.connect('test_database')
    c = conn.cursor()
    #c.execute('CREATE TABLE IF NOT EXISTS trips (region text, datetime number, datasource text, Euclidian_Distance number, origin_x number, origin_y number)')
    conn.commit()
    df.to_sql('trips', conn, if_exists='replace', index = False)