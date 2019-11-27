import streamlit as st


def main():
    st.write('Streamlit and Docker container test')

    sel_option = st.sidebar.selectbox('This is a select box', ['None'] + ['Option A', 'Option B', 'Option C'], index=0)

    if 'None' in sel_option:
        st.write('You have not chosen any option')
    else:
        st.write('You have chosen {}'.format(sel_option))


if __name__ == '__main__':
    main()
