import streamlit as st
import pandas as pd
import numpy as np
import pyodbc
import os
from py_dotenv import read_dotenv
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')) + '\\'

dotenv_path = base_path + 'info.env'
read_dotenv(dotenv_path)


"""
# Streamlit third test
Here's my first attempt at modifying an SQL table
"""

performance_sql_info = {'DSN': os.getenv('DSN_MLG'),
                        'UID': os.getenv('UID'),
                        'PWD': os.getenv('PWD'),
                        'DB': 'BI_MLG',
                        'test_db': 'streamlit_test'
                        # 'log_view': 'LOG_Information',
                        # 'error_log': 'LOG_Performance_Errors',
                        # 'warning_log': 'LOG_Performance_Warnings',
                        # 'model_choices': 'LOG_Performance_Model_Choices',
                        # 'mail_users': 'LOG_MailUsers',
                        # 'performance_running_time': 'LOG_Performance_Running_Time',
                        # 'performance_algorithm_results': 'LOG_Performance_Algorithms_Results',
                        }


def main():

    date = st.date_input("Please select a date:", value=None)
    op_type = st.selectbox("Please select the type of operation:", ['None'] + ['Compra', 'Venda'], index=0)
    value = st.number_input("Please insert a number of rows:")

    cols = ['col1', 'col2', 'col3']

    st.write('Date Selected: {}'.format(date))
    st.write('Operation Type Selected: {}'.format(op_type))
    st.write('Value Selected: {}'.format(value))

    if st.button('Upload to SQL:'):
        if value > 0:
            st.write("Uploading template")
            sql_inject(date, op_type, value, cols)
        else:
            st.write("Please select the number of rows")


def sql_inject(date, op_type, value, cols):

    df = pd.DataFrame(columns=cols)
    df.loc[0, :] = [date, op_type, value]
    columns = list(df)

    columns_string, values_string = sql_string_preparation_v1(cols)

    cnxn = pyodbc.connect('DSN={};UID={};PWD={};DATABASE={}'
                          .format(performance_sql_info['DSN'], performance_sql_info['UID'], performance_sql_info['PWD'], performance_sql_info['DB']), searchescape='\\')

    cursor = cnxn.cursor()
    for index, row in df.iterrows():
        cursor.execute("INSERT INTO " + performance_sql_info['test_db'] + "(" + columns_string + ') VALUES ( ' + '\'{}\', \'{}\', \'{}\''.format(row['col1'], row['col2'], row['col3']) + ' )')

    cursor.commit()
    cursor.close()


def sql_string_preparation_v1(values_list):
    columns_string = '[%s]' % "], [".join(values_list)

    values_string = ['\'?\''] * len(values_list)
    values_string = 'values (%s)' % ', '.join(values_string)

    return columns_string, values_string


if __name__ == '__main__':
    main()
