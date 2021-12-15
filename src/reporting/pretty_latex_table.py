import pandas as pd
import os
def max_values_bold(data, column_max):
    if data == column_max:
        return f"\\textbf{{{data}}}"

    return data

def min_value_bold(data, column_min):
    if data == column_min:
        return  f"\\textbf{{{data}}}"
    return data


def column_header_bold(df: pd.DataFrame) -> list:
    return (df
            .columns.to_series()
            .apply(lambda r: "\\textbf{{{0}}}".format(
            r.replace("_", " ").title())))

def generate_table(df, save_to,filename=None, max_bold=None, min_bold=None, columns_headers_bold=True, caption=None, label=None,column_name_map=None):
    if max_bold:
        for k in max_bold:
            df[k] = df[k].apply(
            lambda data: max_values_bold(data, column_max=df[k].max()))
    if min_bold:
        for k in min_bold:
            df[k] = df[k].apply(
            lambda data: min_value_bold(data, column_min=df[k].min()))

    if column_name_map:
        df.rename(columns=column_name_map,inplace=True)

    if columns_headers_bold:
        df.columns = column_header_bold(df)

    if not filename:
        filename = 'data_tbl.tex'
    if not caption:
        caption = 'Your caption could be here'
    if not label:
        label = 'Your label'

    with open(os.path.join(save_to,filename),"w") as latex_file:

        # format_tbl = "l" + \
        #     "@{\hskip 12pt}" +\
        #     4*"S[table-format = 2.2]"

        latex_file.write(df
                .to_latex(index=False,
                          escape=False,
                          caption=caption,
                          label=label,
                          column_format=f'l*{{{df.shape[1]-1}}}{{c}}').replace('_',' ')

                )




