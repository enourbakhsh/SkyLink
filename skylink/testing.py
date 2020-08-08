import pandas as pd
import astropy
import colored as cl

def standardize(df):
    df_rowsorted = df.sort_values(['row_index','catalog_key'])
    df_rowsorted['group_id'] = pd.factorize(df_rowsorted['group_id'])[0]
    return df_rowsorted.sort_values(['group_id','catalog_key','row_index']).reset_index(drop=True)

def assert_equal_dfs(df1, df2, objtype='dataframes'):
    '''
    Tests whether two sky-matching pandas dataframes are equal
    '''
    ans = standardize(df1).equals(standardize(df2))
    if ans:
        print(cl.stylize(f'✔ The {objtype} are equal!', cl.fg('green')+cl.attr('bold')))
    else:
        print(cl.stylize(f'✘ The {objtype} are not equal!', cl.fg('red')+cl.attr('bold')))

def assert_equal_tables(table1, table2):
    '''
    Tests whether two sky-matching astropy tables are equal
    '''
    df1 = table1.to_pandas()
    df2 = table2.to_pandas()
    assert_equal_dfs(df1, df2, objtype='Tables')

def assert_equal(obj1, obj2):
    '''
    Tests whether two sky-matching objects are equal to each other

    obj1, obj2: pandas dataframes or astropy Tables, if any of them is a
                path (string) to the corresponding pandas pickle,
                we will read that pickle for you first!
    '''
    if isinstance(obj1, str):
        print('- Loading pandas pickle for the first object ...')
        obj1 = pd.read_pickle(obj1)

    if isinstance(obj2, str):
        print('- Loading pandas pickle for the second object ...')
        obj2 = pd.read_pickle(obj2)

    if type(obj1) != type(obj2):
        raise TypeError('The two objects must have the same type, either pass two pandas dataframes '
                        '(or path to one or two pickles) or two astropy Tables')
    if type(obj1) == pd.DataFrame:
        assert_equal_dfs(obj1, obj2)
    elif type(obj1) == astropy.table.Table:
        assert_equal_tables(obj1, obj2)
