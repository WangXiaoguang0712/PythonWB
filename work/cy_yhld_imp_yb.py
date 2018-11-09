# _*_ coding:utf-8 _*_
import cx_Oracle as ora
import xlrd
import os
import chardet

# for python 2.7  32bit
def load_data(sheet_name, tab_name, cols, filename):
    path_excel = u'D:\\项目文件\\朝阳医院\\统计数据\\' + filename
    tns = ora.makedsn('172.16.19.71', 1521, 'OCSDB')
    print(path_excel)


    wb = xlrd.open_workbook(path_excel)
    ws = wb.sheet_by_name(sheet_name)

    sql = "insert all "
    print(ws.nrows)
    for i in range(1, ws.nrows):
        sql_values = "into "+ tab_name +" values("
        for j in range(cols):
            val = ws.row_values(i)[j]
            if j == 2 or j == 3:
                sql_values += "to_date('"+ val.rstrip(' ') +"','yyyy-mm-dd'),"
            else:
                if j == 1 and len(str(val).rstrip(' ')) > 20:
                    print(val)
                    val = val[:20]
                if isinstance(val, float):
                    val = str(val)

                sql_values += "'"+ val.rstrip(' ') +"',"
                if j == cols - 1:
                    sql_values = sql_values.rstrip(',')
        sql_values += ") "
        sql += sql_values
        print(i)
        if i % 2000 == 0 or i == ws.nrows - 1:
            sql += "select 1 from dual "
            print(sql.encode('gbk'))
            conn = ora.connect('cy_hsp_spk', 'SPK', tns)
            cur = conn.cursor()
            cur.execute(sql.encode('gbk'))
            cur.close()
            conn.commit()
            conn.close()
            sql = "insert all "



def select_db():
    tns = ora.makedsn('172.16.19.71', 1521, 'OCSDB')
    conn = ora.connect('cy_hsp_spk', 'SPK', tns)
    cur = conn.cursor()
    # sql = "delete from YHLD_imp_ybop"
    sql = 'select count(*) from YHLD_imp_ybip '
    #sql = "insert all into YHLD_imp_ybop values('3','120100012','造口护理','214.0','1070.0') " \
    #      "into YHLD_imp_ybop values ('3','250311007','胶原降解产物测定','236.0','25960.0') " \
    #      "select 1 from dual"
    cur.execute(sql)
    for i in cur.fetchall():
        print(i)
    cur.close()
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # select_db()
    # load_data('ip_w', 'YHLD_imp_ybip', 5, u'17年医保明细.xls')
    # load_data('ip_w', 'YHLD_imp_ybip')
    # load_data('op_w', 'YHLD_icd_ybop2', 6, '17年医保明细')
    load_data('ip_person', 'yhld_ip_yb_person', 5, u'17年医保明细.xls')