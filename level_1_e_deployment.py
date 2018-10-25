import os
import time
import pyodbc
import pandas as pd


def save_csv(dfs, names):
    # Checks for file existence and deletes it if exists, then saves it

    for i, df in enumerate(dfs):
        name = names[i] + '.csv'
        if os.path.isfile(name):
            os.remove(name)
        df.to_csv(name)


def sql_inject(df, database, view):
    start = time.time()

    print('Uploading to SQL Server to DB ' + database + ' and view ' + view + '...')

    cnxn = pyodbc.connect('DSN=MLG;UID=interplataformas;PWD=inf2008;DATABASE=' + database)
    cursor = cnxn.cursor()

    for index, row in df.iterrows():
        cursor.execute("INSERT INTO " + view + "("
                                               "[Auto_Trans],[Navigation],[Park_Front_Sens],[Rims_Size],[Colour_Int],[Colour_Ext],[Sales_Place], "
                                               "[Model_Code], [Purchase_Day], [Purchase_Month], [Purchase_Year], [Source_Desc],[Margin], [Margin_Percentage], "
                                               "[Stock_Days_Price], [Score_Euros], [Stock_Days],[Sell_Value], [Probability_0], [Probability_1], [Score_Class_GT], "
                                               "[Score_Class_Pred], [Sell_Date])"
                       "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       row['Auto_Trans'], row['Navigation'], row['Park_Front_Sens'], row['Rims_Size'], row['Colour_Int'], row['Colour_Ext'],
                       row['Sales_Place'], row['Model_Code'], row['Purchase_Day'], row['Purchase_Month'], row['Purchase_Year'], row['Source_Desc'],
                       row['Margin'], row['Margin_Percentage'], row['Stock_Days_Price'], row['Score_Euros'], row['Stock_Days'],
                       row['Sell_Value'], row['Probability_0'], row['Probability_1'], row['Score_Class_GT'], row['Score_Class_Pred'], row['Sell_Date'])

    cnxn.commit()
    cursor.close()
    cnxn.close()

    print('Elapsed time: %.2f' % (time.time() - start), 'seconds...')


def sql_truncate(database, view):
    print('Truncating view ' + view + ' from DB ' + database)
    cnxn = pyodbc.connect('DSN=MLG;UID=interplataformas;PWD=inf2008;DATABASE=' + database)
    # query = "SELECT * FROM " + database + ".dbo." + view
    query = "TRUNCATE TABLE " + view
    cursor = cnxn.cursor()
    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()

