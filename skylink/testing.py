import pandas as pd
import astropy
import colored as cl


def standardize(df):
    """Converts the input dataframe to a standard dataframe in terms of group
    ids and row ordering. This help with dataframe comparisons.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        Input dataframe.

    Returns
    -------
    `~pandas.DataFrame`
        The standardized dataframe.

    """
    df_rowsorted = df.sort_values(["row_index", "catalog_key"])
    df_rowsorted["group_id"] = pd.factorize(df_rowsorted["group_id"])[0]
    return df_rowsorted.sort_values(
        ["group_id", "catalog_key", "row_index"]
    ).reset_index(drop=True)


def assert_equal_dfs(df1, df2, objtype="dataframes", raise_error=True):
    """Tests whether two sky-matching pandas dataframes are equal.

    Parameters
    ----------
    df1 : `~pandas.DataFrame`
        The first dataframe.
    df2 : `~pandas.DataFrame`
        The second dataframe.
    objtype : str, optional, default: 'dataframes'
        The type of the original input files in comparison.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the two dataframes are not equal.

    """
    ans = standardize(df1).equals(standardize(df2))
    if ans:
        print(
            cl.stylize(f"✔ The {objtype} are equal catalogs!", cl.fg("green") + cl.attr("bold"))
        )
    else:
        message = cl.stylize(f"✘ The {objtype} are not equal catalogs!", cl.fg("red") + cl.attr("bold"))
        if raise_error:
            raise RuntimeError(message)
        else:
            print(message)


def assert_equal_tables(table1, table2, raise_error=True):
    """Tests whether two sky-matching astropy tables are equal.

    Parameters
    ----------
    table1 : `~astropy.table.Table`
        The first table.
    table2 : `~astropy.table.Table`
        The second table.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the two tables are not equal.

    """
    df1 = table1.to_pandas()
    df2 = table2.to_pandas()
    assert_equal_dfs(df1, df2, objtype="Tables", raise_error=raise_error)


def assert_equal(obj1, obj2, raise_error=True):
    """Tests whether two sky-matching objects are equal to each other.

    obj1, obj2: pandas dataframes or astropy Tables, if any of them is a
                path (string) to the corresponding pandas pickle,
                we will read that pickle for you first!

    Parameters
    ----------
    obj1 : str or `~pandas.DataFrame` or `~astropy.table.Table`
        The first table or dataframe. If a path (string) to the corresponding
        pandas pickle is provided, it will read the dataframe from that pickle.
    obj2 : str or `~pandas.DataFrame` or `~astropy.table.Table`
        The second table or dataframe. If a path (string) to the corresponding
        pandas pickle is provided, it will read the dataframe from that pickle.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the two objects do not have the same type or are not pandas
        dataframes, paths to pandas dataframe pickles, or astropy Tables.

    RuntimeError
        If the two dataframes or tables are not equal.

    """
    if isinstance(obj1, str):
        print("- Loading pandas pickle for the first object ...")
        obj1 = pd.read_pickle(obj1)

    if isinstance(obj2, str):
        print("- Loading pandas pickle for the second object ...")
        obj2 = pd.read_pickle(obj2)

    if not isinstance(obj1, type(obj2)):
        raise TypeError(
            "The two objects must have the same type, either pass two pandas dataframes "
            "(or path to one or two pickles) or two astropy Tables"
        )
    if isinstance(obj1, pd.DataFrame):
        assert_equal_dfs(obj1, obj2, raise_error=raise_error)
    elif isinstance(obj1, astropy.table.Table):
        assert_equal_tables(obj1, obj2, raise_error=raise_error)
    else:
        raise TypeError(
            "The input files should be pandas dataframes (or path to one or two pickles) or astropy Tables"
        )
