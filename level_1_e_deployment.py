import os
import time
import pyodbc
import sys


def save_csv(dfs, names):
    # Checks for file existence and deletes it if exists, then saves it

    for i, df in enumerate(dfs):
        name = names[i] + '.csv'
        if os.path.isfile(name):
            os.remove(name)
        df.to_csv(name)


def sql_inject(df, database, view, columns):
    start = time.time()
    columns_string = str()
    values_string = 'values ('

    print('Uploading to SQL Server to DB ' + database + ' and view ' + view + '...')

    cnxn = pyodbc.connect('DSN=MLG;UID=interplataformas;PWD=inf2008;DATABASE=' + database)
    cursor = cnxn.cursor()

    for item in list(columns.values()):
        columns_string += '[' + item + '], '
        values_string += '?, '

    columns_string = columns_string[:-2] + '' + columns_string[-1:]
    values_string = values_string[:-2] + ')' + values_string[-1:]

    for index, row in df.iterrows():
        cursor.execute("INSERT INTO " + view + "(" + columns_string + ') ' + values_string, [row[value] for value in list(columns.values())])

    # for index, row in df.iterrows():
    #     cursor.execute("INSERT INTO " + view + "("
    #                                            "[Auto_Trans], [Navigation], [Park_Front_Sens], [Rims_Size], [Colour_Int], [Colour_Ext], [Sales_Place], "
    #                                            "[Model_Code], [Purchase_Day], [Purchase_Month], [Purchase_Year], [Margin], [Margin_Percentage], "
    #                                            "[Stock_Days_Price], [Score_Euros], [Stock_Days], [Sell_Value], [Probability_0], [Probability_1], [Score_Class_GT], "
    #                                            "[Score_Class_Pred], [Sell_Date], [Seven_Seats], [AC_Auto], [Alarm], [Roof_Bars], [Open_Roof], [LED_Lights], "
    #                                            "[Xenon_Lights], [Solar_Protection], [Interior_Type], [Version], [Average_Margin_Percentage], [Average_Score_Euros],  "
    #                                            "[Average_Stock_Days], [Average_Score_Class_GT], [Average_Score_Class_Pred], [Number_Cars_Sold], [Number_Cars_Sold_Local],"
    #                                            "[Average_Margin_Percentage_Local], [Average_Score_Euros_Local], [Average_Stock_Days_Local], [Average_Score_Class_GT_Local], "
    #                                            "[Average_Score_Class_Pred_Local])"
    #                    "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    #                    row['Auto_Trans'], row['Navigation'], row['Park_Front_Sens'], row['Rims_Size'], row['Colour_Int'], row['Colour_Ext'],
    #                    row['Sales_Place'], row['Model_Code'], row['Purchase_Day'], row['Purchase_Month'], row['Purchase_Year'],
    #                    row['Margin'], row['Margin_Percentage'], row['Stock_Days_Price'], row['Score_Euros'], row['Stock_Days'],
    #                    row['Sell_Value'], row['Probability_0'], row['Probability_1'], row['Score_Class_GT'], row['Score_Class_Pred'], row['Sell_Date'],
    #                    row['Seven_Seats'], row['AC_Auto'], row['Alarm'], row['Roof_Bars'], row['Open_Roof'], row['LED_Lights'], row['Xenon_Lights'], row['Solar_Protection'],
    #                    row['Interior_Type'], row['Version'], row['Average_Margin_Percentage'], row['Average_Score_Euros'],  row['Average_Stock_Days'], row['Average_Score_Class_GT'],
    #                    row['Average_Score_Class_Pred'], row['Number_Cars_Sold'], row['Number_Cars_Sold_Local'], row['Average_Margin_Percentage_Local'], row['Average_Score_Euros_Local'],
    #                    row['Average_Stock_Days_Local'], row['Average_Score_Class_GT_Local'], row['Average_Score_Class_Pred_Local'])

    cnxn.commit()
    cursor.close()
    cnxn.close()

    print('Elapsed time: %.2f' % (time.time() - start), 'seconds...')


def sql_truncate(database, view):
    print('Truncating view ' + view + ' from DB ' + database)
    cnxn = pyodbc.connect('DSN=MLG;UID=interplataformas;PWD=inf2008;DATABASE=' + database)
    query = "TRUNCATE TABLE " + view
    cursor = cnxn.cursor()
    cursor.execute(query)

    cnxn.commit()
    cursor.close()
    cnxn.close()
