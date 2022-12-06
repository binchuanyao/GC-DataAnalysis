# -*- coding: utf-8 -*-
# @File   : interface
# @Time   : 2022/07/26 10:43 
# @Author : BCY

import os.path

import numpy as np
import pandas as pd
import datetime
from config import *
import resource

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QStatusBar
from PyQt5.QtCore import QAbstractTableModel, Qt
from data_analysis import Ui_MainWindow


# import matplotlib
# matplotlib.use("Qt5Agg")  # 声明使用QT5

# 实例化配置参数, 全局变量
config = Config()
config.run()

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle('GC数据分析')

        self.statusbar.showMessage('谷仓-仓储管理部-规划实施组')
        self.statusbar.addPermanentWidget(QLabel("数据分析V1.3"), stretch=0)  # 比例
        # self.statusBar.addPermanentWidget(self.show_2, stretch=0)

        '''基础数据dataframe'''
        self.inv_df = pd.DataFrame()
        self.sku_df = pd.DataFrame()
        self.ob_df = pd.DataFrame()       # 出库明细原始数据
        self.ob_time_df = pd.DataFrame()  # 订单流入时间数据
        self.ob_validDF = pd.DataFrame()  # 用于存储出库数据处理后的有效数据
        self.ib_df = pd.DataFrame()

        '''库存分析结果dataframe变量'''
        self.design_location_df = pd.DataFrame()
        self.current_location_df = pd.DataFrame()
        self.inventory_customer_df = pd.DataFrame()
        self.inventory_age_df = pd.DataFrame()
        self.inv_info = ''

        '''出库分析结果deatframe变量，设置为类的属性以便在保存时调用'''
        ### EIQ分析结果
        self.normal_order_EIQ = pd.DataFrame()
        self.fba_order_EIQ = pd.DataFrame()
        self.order_type_EIQ = pd.DataFrame()
        self.wave_order_type_EIQ = pd.DataFrame()
        self.current_pick_order_EIQ = pd.DataFrame()
        self.multi_order_EIQ = pd.DataFrame()

        ### 客户动销分析
        self.customer_sales = pd.DataFrame()

        ### sku动销分析
        self.sku_active_sales = pd.DataFrame()

        ### 库龄分析
        self.sku_age_df = pd.DataFrame()
        self.order_age_df = pd.DataFrame()

        ### 小时订单流入
        self.sku_hour_in_df = pd.DataFrame()

        ### ABC分析
        self.sku_ABC_df = pd.DataFrame()
        self.sku_ABC_detail = pd.DataFrame()
        self.abc_info = ''

        ### 库龄分析
        self.sku_age_df = pd.DataFrame()
        self.order_age_df = pd.DataFrame()

        ### 渠道分析
        self.channel_order_distribution = pd.DataFrame()

        ### 平台分析
        self.platform_order_distribution =pd.DataFrame()


        '''入库分析结果deatframe变量，设置为类的属性以便在保存时调用'''
        self.ib_date_distribution = pd.DataFrame()         # 日来货分布
        self.ib_container_carton_type = pd.DataFrame()     # 海柜来货箱型分布
        self.ib_container_skuNum = pd.DataFrame()          # 海柜sku种类分布
        self.ib_daily_container_num = pd.DataFrame()       # 日来柜数量
        self.ib_daily_container_sku = pd.DataFrame()       # 日来柜SKU数
        self.ib_info = ''



        ### 绑定 按钮 与 事件函数
        '''库存分析'''

        self.inv_upload.clicked.connect(self.inventory_load_data)  # 导入库存数据
        self.inv_uploadSKU.clicked.connect(self.load_sku_data)     # 导入sku基础数据

        self.inv_dataProcess.clicked.connect(self.inventory_data_process)   # 数据处理
        self.inv_inventoryAnalysis.clicked.connect(self.inventory_analysis)  # 库存分析

        self.inv_clear.clicked.connect(self.inventory_clear_data)   # 清除所有库存数据
        self.inv_download.clicked.connect(self.inventory_save_data)


        '''出库分析'''
        self.ob_orderUpload.clicked.connect(self.outbound_load_order_data)           # 导入出库数据
        self.ob_orderTimeUpload.clicked.connect(self.outbound_load_order_time_data)  # 导入出库时间
        self.ob_dataProcess.clicked.connect(self.outbound_data_process)              # 数据处理


        '''出库分析维度'''
        self.ob_EIQanalysis.clicked.connect(self.outbound_EIQ_analysis)   # EIQ分析
        self.ob_multiOrder.clicked.connect(self.outbound_multi_order_analysis)   # 多品订单组合分析
        self.ob_customerSale.clicked.connect(self.outbound_customer_sale_analysis)  # 客户动销分析
        self.ob_skuSale.clicked.connect(self.outbound_sku_active_sale_analysis)  # sku动销分析

        self.ob_skuAge.clicked.connect(self.outbound_sku_age_analysis)  # sku库龄分析
        self.ob_orderAge.clicked.connect(self.outbound_order_age_analysis)  # 订单库龄分析
        self.ob_skuHourIn.clicked.connect(self.outbound_sku_hour_in_analysis)  # sku小时流入分析
        self.ob_ABCanalysis.clicked.connect(self.outbound_ABC_analysis)  # ABC分析

        self.ob_channelAnalysis.clicked.connect(self.outbound_channel_analysis)  # 渠道分析
        self.ob_platformAnalysis.clicked.connect(self.outbound_platform_analysis)  # 平台分析
        self.ob_downloadAll.clicked.connect(self.outbound_download_all_results)  # 导出所有分析
        self.ob_clearAll.clicked.connect(self.outbound_clear_all)  # 清楚所有数据


        '''入库分析'''
        self.ib_upload.clicked.connect(self.inbound_load_data)          # 导入入库数据
        self.ib_dataProcess.clicked.connect(self.inbound_data_process)  # 入库数据处理
        self.ib_inboundAnalysis.clicked.connect(self.inbound_analysis)  # 来货量分析
        self.ib_cartonDistribution.clicked.connect(self.inbound_carton_distribution)    # 箱数分布
        self.ib_download.clicked.connect(self.inbound_download_all)     # 入库数据导出
        self.ib_clear.clicked.connect(self.inbound_clear_all)           # 入库数据重置


    def show_error_dialog(self, msg):
        QMessageBox.critical(self, '错误', msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def show_info_dialog(self, msg):
        QMessageBox.information(self, '消息', msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def resource_path(self, relative_path):
        '''将相对路径转为运行时资源文件的绝对路径'''
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath('.')
        return os.path.join(base_path, relative_path)


    def inventory_load_data(self):
        try:
            filenames = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls, *.csv)')
            filename = filenames[0]

            if 'csv' in filename:
                try:
                    self.inv_df = pd.read_csv(filename, encoding='utf-8')
                except:
                    self.inv_df = pd.read_csv(filename, encoding='gbk')
                # 删除有空值的行
                self.inv_df.dropna(how='all', inplace=True)
                model = dfModel(self.inv_df.head(100))
                self.inv_tableView.setModel(model)
            elif 'xlsx' in filename:
                self.inv_df = pd.read_excel(filename)
                # 删除有空值的行
                self.inv_df.dropna(how='all', inplace=True)
                model = dfModel(self.inv_df.head(100))
                self.inv_tableView.setModel(model)
            else:
                self.show_error_dialog('请选择csv或xlsx文件类型!')
        except:
            self.show_error_dialog('请选择文件!')


    def load_sku_data(self):
        try:
            filenames = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls, *.csv)')
            filename = filenames[0]

            if 'csv' in filename:
                try:
                    self.sku_df = pd.read_csv(filename, encoding='utf-8')
                except:
                    self.sku_df = pd.read_csv(filename, encoding='gbk')
                # 删除有空值的行
                self.sku_df.dropna(how='all', inplace=True)
                model = dfModel(self.sku_df.head(100))
                self.sku_tableView.setModel(model)

            elif 'xlsx' in filename:
                self.sku_df = pd.read_excel(filename)
                # 删除有空值的行
                self.sku_df.dropna(how='all', inplace=True)

                print(self.sku_df.head(10))

                model = dfModel(self.sku_df.head(100))
                print(model)
                self.sku_tableView.setModel(model)

            else:
                self.show_error_dialog('请选择csv或xlsx文件类型!')
        except:
            self.show_error_dialog('导入SKU数据错误！')

    def inventory_data_process(self):
        try:
            # 输入库存数据对应列编号
            inv_dateCol = self.inv_df.columns[int(self.inv_dateCol.text())]  # 库存日期
            inv_putawayDateCol = self.inv_df.columns[int(self.inv_putawayDateCol.text())]  # 上架日期
            inv_customerCol = self.inv_df.columns[int(self.inv_customerCol.text())]  # 客户代码
            inv_skuCol = self.inv_df.columns[int(self.inv_skuCol.text())]    # 产品代码
            inv_skuSizeCol = self.inv_df.columns[int(self.inv_skuSizeCol.text())]    # 产品货型
            inv_invQty = self.inv_df.columns[int(self.inv_invQty.text())]    # 在库件数
            inv_invVol = self.inv_df.columns[int(self.inv_invVol.text())]    # 在库体积

            inv_invTypeCol = self.inv_df.columns[int(self.inv_locTypeCol.text())]    # 储位类型
            inv_skuLengthCol = self.inv_df.columns[int(self.inv_skuLengthCol.text())]    # 产品长
            inv_skuWidthCol = self.inv_df.columns[int(self.inv_skuWidthCol.text())]    # 产品宽
            inv_skuHeightCol = self.inv_df.columns[int(self.inv_skuHeightCol.text())]    # 产品高
            inv_locNumCol = self.inv_df.columns[int(self.inv_locNumCol.text())]  # 库位数量


            self.inv_info = '物理仓名称：{}'.format(self.inv_df['物理仓名称'].unique()[0])

            valid_columns = [inv_dateCol, inv_putawayDateCol, inv_customerCol, inv_skuCol, inv_skuSizeCol, inv_invQty, inv_invVol,
                  inv_invTypeCol, inv_skuLengthCol, inv_skuWidthCol, inv_skuHeightCol, inv_locNumCol]

            print('输入的列编号对应的列名：')
            print(valid_columns)

            self.inv_df = self.inv_df[valid_columns]    # 保留有效字段
            self.inv_df.dropna(how='any', inplace=True)   # 删除有缺失的行

            # 返回指定日期的批次库存数据,否则返回所有数据
            d_date = self.inv_dateEdit.text()
            all_invDates = self.inv_df[inv_dateCol].unique()

            print('d_date:', d_date)
            print('all_invDates:', all_invDates)
            print( d_date in all_invDates)

            if d_date in all_invDates:
                self.inv_df = self.inv_df.where(self.inv_df[inv_dateCol] == d_date)
                self.inv_info = self.inv_info + '; 批次库存日期：{}'.format(d_date)
            else:
                self.inv_info = self.inv_info + '; 批次库存日期：{}'.format(self.inv_df[inv_dateCol].unique())

            # 计算库龄
            self.inv_df[inv_dateCol] = pd.to_datetime(self.inv_df[inv_dateCol])
            self.inv_df[inv_putawayDateCol] = pd.to_datetime(self.inv_df[inv_putawayDateCol])
            self.inv_df['库龄'] = pd.to_timedelta(self.inv_df[inv_dateCol] - self.inv_df[inv_putawayDateCol]).dt.days

            ## 计算库位维度库龄类别   AGE_CLASS 数据结构为[['D1(0,30]', 0, 30], ['D2(30,60]', 30, 60]] 二维列表，其中的元素为【等级，左区间，右区间】
            self.inv_df['库龄等级'] = ''
            rank_num = len(config.AGE_CLASS)
            for i in range(rank_num):
                self.inv_df.loc[(self.inv_df['库龄'] > config.AGE_CLASS[i][1]) & (self.inv_df['库龄'] <= config.AGE_CLASS[i][2]), ['库龄等级']] = config.AGE_CLASS[i][0]

            print('库存数据处理完成！')
            # 输入sku对应列编号
            sku = self.sku_df.columns[int(self.skuCol.text())]
            skuSizeCol = self.sku_df.columns[int(self.skuSizeCol.text())]
            skuLengthCol = self.sku_df.columns[int(self.skuLengthCol.text())]
            skuWidthCol = self.sku_df.columns[int(self.skuWidthCol.text())]
            skuHeightCol = self.sku_df.columns[int(self.skuHeightCol.text())]

            cols = [sku, skuSizeCol, skuLengthCol, skuWidthCol, skuHeightCol]
            self.sku_df = self.sku_df[cols].drop_duplicates()

            print('sku数据处理完成！')

            '''显示数据预览'''
            row1 = self.inv_df.shape[0]
            col1 = self.inv_df.shape[1]
            skuNum = self.sku_df.shape[0]
            print('库存数据预览：', row1, col1)
            self.inv_info = self.inv_info + '; 库存数据量：行数 {}, 列数 {}'.format(row1, col1)

            row2 = self.sku_df.shape[0]
            col2 = self.sku_df.shape[1]
            print('SKU数据预览：', row2)
            self.inv_info = self.inv_info + '; SKU数据量：行数 {}, 列数 {}'.format(row2, col2)

            # 去重
            self.inv_df = self.inv_df.drop_duplicates()
            self.inv_totalRow.setText("{:,}".format(row1))  #数据总行数
            self.inv_totalCol.setText("{:,}".format(col1))  #数据总列数

            self.inv_errorRow.setText("{:,}".format(row1-row2))  #重复行数
            self.inv_validRow.setText("{:,}".format(row2))       #数据去重后总行数

            self.inv_totalSKU.setText("{:,}".format(skuNum))
            self.inv_totalDays.setText("{:,}".format(self.inv_df[inv_dateCol].nunique()))

            print('数据预览完成！')

            # 刷新库存数据
            self.inv_tableView.setModel(dfModel(self.inv_df.head(100)))
            self.show_info_dialog('库存【数据处理】完成！')

        except:
            self.show_error_dialog('请输入对应字段列编号！')

    def inventory_analysis(self):

        try:
            print('in inventory_analysis！')
            print(self.inv_df.columns)
            # 输入库存数据对应列编号
            [inv_dateCol, inv_putawayDateCol, inv_customerCol, inv_skuCol, inv_skuSizeCol, inv_invQty, inv_invVol,
             inv_invTypeCol, inv_skuLengthCol, inv_skuWidthCol, inv_skuHeightCol, inv_locNumCol] = self.inv_df.columns[0:12]

            print('输入的列编号对应的列名：')
            print(inv_dateCol, inv_skuCol, inv_customerCol, inv_putawayDateCol, inv_skuSizeCol, inv_invQty, inv_invVol)


            # 简单库位匹配，一个sku只匹配一种库位类型
            self.design_location_df, self.current_location_df = self.inventory_calculate_single_location()

            print('库位类型计算完成!!!')


            '''客户维度分析'''
            self.inventory_customer_df = self.inventory_customer_pivot()

            '''
            库龄等级分析
            '''
            df_sku = self.inventory_get_sku_pivot(inv_skuCol, inv_invQty, inv_invVol)  # sku维度库龄
            sku_col = [inv_skuCol, inv_invQty, inv_invVol, '库龄', '库龄等级']  # 重命名列
            df_sku.columns = sku_col

            df_age = pd.pivot_table(df_sku, index=['库龄等级'],
                                    values=[inv_invVol,  inv_invQty, inv_skuCol],
                                    aggfunc={inv_invVol: np.sum, inv_invQty: np.sum, inv_skuCol: len},
                                    margins=True,
                                    margins_name='合计',
                                    fill_value=0).reset_index()

            ## 列的重命名 透视表中排序以拼音首字母顺序
            new_columns = ['库龄等级', '在库sku数', '在库件数', '在库体积(m³)']
            df_age.columns = new_columns

            ## 计算库存体积和件数的占比
            for col in new_columns[1:]:
                df_age[col + '%'] = df_age[col] / (df_age[col].sum() / 2)

            df_age['库存深度(m³/sku)'] = df_age['在库体积(m³)'] / df_age['在库sku数']
            df_age['库存深度(件/sku)'] = df_age['在库件数'] / df_age['在库sku数']

            self.inventory_age_df = df_age
            self.sku_tableView.setModel(dfModel(self.inventory_age_df))
            self.show_info_dialog('库存分析计算完成！')

        except:
            self.show_error_dialog('【库存分析】计算出错！！！')

    def inventory_calculate_single_location(self):
        """
        :param df: 透视表原始数据
        :return:
        """

        '''库存数据基础字段计算'''
        try:
            print('in calculate_single_location')
            # inv_invVol = self.inv_df.columns[int(self.inv_invVol.text())]  # 在库体积
            # inv_skuSizeCol = self.inv_df.columns[int(self.inv_skuSizeCol.text())]  # 产品货型
            # inv_skuLengthCol = self.inv_df.columns[int(self.inv_skuLengthCol.text())]  # 产品长

            # 库存数据中的列名
            [inv_dateCol, inv_putawayDateCol, inv_customerCol, inv_skuCol, inv_skuSizeCol, inv_invQty, inv_invVol,
             inv_invTypeCol, inv_skuLengthCol, inv_skuWidthCol, inv_skuHeightCol, inv_locNumCol] = self.inv_df.columns[0:12]


            config = Config()
            config.run()

            self.inv_df['在库托数'] = self.inv_df[inv_invVol] / config.PALLET['valid_vol']

            self.inv_df['是否超长'] = 'N'
            self.inv_df.loc[(self.inv_df[inv_skuLengthCol] >= config.SUPER_LONG_PARAM), ['是否超长']] = 'Y'

            self.inv_df['是否批量'] = 'N'
            self.inv_df.loc[(self.inv_df['在库托数'] >= config.BATCH_PARAM), ['是否批量']] = 'Y'

            self.inv_df['储位类型'] = np.NAN
            self.inv_df.loc[(self.inv_df['储位类型'].isna()) & (self.inv_df['是否批量'] == 'Y'), ['储位类型']] = '批量平铺区'
            self.inv_df.loc[(self.inv_df['储位类型'].isna()) & (self.inv_df['是否超长'] == 'Y'), ['储位类型']] = '超长平铺区'
            self.inv_df.loc[(self.inv_df['储位类型'].isna()) & (self.inv_df['是否超长'] == 'N') & (self.inv_df[inv_skuSizeCol] == 'XL'), ['储位类型']] = '异形高架区'
            self.inv_df.loc[(self.inv_df['储位类型'].isna()) & (self.inv_df[inv_skuSizeCol] == 'L1') | (self.inv_df[inv_skuSizeCol] == 'L2'), ['储位类型']] = '卡板区'
            self.inv_df.loc[(self.inv_df['储位类型'].isna()) & (self.inv_df[inv_invVol] >= config.PALLET['min_vol']), ['储位类型']] = '卡板区'
            self.inv_df.loc[(self.inv_df['储位类型'].isna()) & (self.inv_df[inv_invVol] >= config.BOX['min_vol']), ['储位类型']] = '原箱区'
            self.inv_df.loc[(self.inv_df['储位类型'].isna()), ['储位类型']] = '储位盒区'

            self.inv_df['储区类型'] = self.inv_df['储位类型']
            self.inv_df.loc[(self.inv_df['储位类型'] =='卡板区') & (self.inv_df['在库托数']==1), ['储区类型']] = '单卡板区'
            self.inv_df.loc[(self.inv_df['储位类型'] == '卡板区') & (self.inv_df['在库托数'] >= config.PALLET_INTERVAL[2]) & (self.inv_df['在库托数'] <= 2), ['储区类型']] = '单卡板区'

            pltClassNum = len(config.PALLET_CLASS)
            for i in range(pltClassNum):
                self.inv_df.loc[(self.inv_df['储位类型'] == '卡板区') &
                       (self.inv_df['在库托数'] > config.PALLET_CLASS[i][1]) &
                       (self.inv_df['在库托数'] <= config.PALLET_CLASS[i][2]),
                       ['储区类型']] = config.PALLET_CLASS[i][0]

            # 按标准库位类型匹配的设计库位 , A1.1
            design_location_pt = self.inventory_design_location()
            # print('返回的design_location_pt: ', design_location_pt)

            '''新增现有批次库存货位类型分布'''
            current_location_pt = self.inventory_current_location()

            # print('返回的current_location_pt: ', current_location_pt)
            return design_location_pt, current_location_pt
        except:
            self.show_error_dialog('【库存分析】储位类型维度计算出错！！！！！！')

    def inventory_design_location(self, index=None, pt_col=None):
        try:
            # inv_dateCol = self.inv_df.columns[int(self.inv_dateCol.text())]  # 库存日期
            # inv_skuCol = self.inv_df.columns[int(self.inv_skuCol.text())]  # 产品代码
            # inv_invQty = self.inv_df.columns[int(self.inv_invQty.text())]  # 在库件数
            # inv_invVol = self.inv_df.columns[int(self.inv_invVol.text())]  # 在库体积

            # 库存数据中的列名
            [inv_dateCol, inv_putawayDateCol, inv_customerCol, inv_skuCol, inv_skuSizeCol, inv_invQty, inv_invVol,
             inv_invTypeCol, inv_skuLengthCol, inv_skuWidthCol, inv_skuHeightCol, inv_locNumCol] = self.inv_df.columns[0:12]

            if index is None:
                index = [inv_dateCol, '储位类型']

            if pt_col is None:
                pt_col = [inv_skuCol, inv_invQty, inv_invVol]

            # datetime64[ns]不能作为key,将日期列的格式转换为string
            self.inv_df[inv_dateCol] = self.inv_df[inv_dateCol].astype(str)


            df_location = pd.pivot_table(self.inv_df, index=index,
                                         values=pt_col,
                                         aggfunc={inv_skuCol: pd.Series.nunique, inv_invQty: np.sum, inv_invVol: np.sum},
                                         fill_value=0).reset_index()

            # 重排列
            re_pt_col = [inv_invVol, inv_invQty, inv_skuCol]
            df_location = df_location[index + re_pt_col]

            # 重命名列
            df_location.columns = index + ['在库体积(m³)', '在库件数', 'sku数']

            # 透视字段， 需计算总和及百分比的字段
            sum_col = ['在库体积(m³)', '在库件数', 'sku数']

            # 计算库存深度
            df_location['库存深度(m³/sku)'] = df_location['在库体积(m³)'] / df_location['sku数']
            df_location['库存深度(件/sku)'] = df_location['在库件数'] / df_location['sku数']

            '''
            合并库位类型与透视结果
            '''
            # print('库位类型： \n')
            # print(config.LOCATION_DF)
            # print('\n')

            df_location = pd.merge(config.LOCATION_DF, df_location, on='储位类型', how='right', sort=False).fillna(0)
            # print('df_location: \n', df_location)

            # 设计参数
            df_location['单sku需求库位数'] = df_location['库存深度(m³/sku)'] / df_location['有效库容(m³)']
            df_location['库位需求1-库容最大化'] = np.ceil(df_location['在库体积(m³)'] / df_location['有效库容(m³)'])
            df_location['库位需求2-sku数最大化'] = np.ceil(df_location['sku数'] * df_location['单sku需求库位数'] / df_location['sku限制'])
            df_location['库位需求-现有库存'] = df_location[['库位需求1-库容最大化', '库位需求2-sku数最大化']].max(axis=1)
            df_location['库容饱和系数'] = 0.7
            df_location['规划库位需求数量'] = df_location['库位需求-现有库存'] / df_location['库容饱和系数']
            df_location['面积需求'] = df_location['在库体积(m³)'] / df_location['库容坪效系数']

            row_n = df_location.shape[0]
            # 更新合计值
            df_location.at[row_n, index[-1:]] = '合计'
            df_location.at[row_n, sum_col] = df_location[sum_col].apply(lambda x: x.sum())

            # 计算比例
            for i in range(len(sum_col)):
                df_location[sum_col[i] + '%'] = df_location[sum_col[i]] / (df_location[sum_col[i]].sum() / 2)

            df_location['库存深度(m³/sku)'] = df_location['在库体积(m³)'] / df_location['sku数']
            df_location['库存深度(件/sku)'] = df_location['在库件数'] / df_location['sku数']

            sum_design_col = ['库位需求1-库容最大化', '库位需求2-sku数最大化', '库位需求-现有库存', '规划库位需求数量', '面积需求']
            df_location.at[row_n, sum_design_col] = df_location[sum_design_col].apply(lambda x: x.sum())

            '''
            库位需求结果 列重排
            '''

            re_columns = ['储位类型', '长(cm)', '宽(cm)', '高(cm)', '库容利用率', '有效库容(m³)', 'sku限制', '库容坪效系数',
                          '批次库存日期', '在库体积(m³)', '在库件数', 'sku数', '在库体积(m³)%', '在库件数%', 'sku数%',
                          '库存深度(m³/sku)', '库存深度(件/sku)', '单sku需求库位数',
                          '库位需求1-库容最大化', '库位需求2-sku数最大化', '库位需求-现有库存', '库容饱和系数', '规划库位需求数量', '面积需求']

            df_location = df_location[re_columns]
            # print('设计储位类型：\n', df_location)

            return df_location
        except:
            self.show_error_dialog('【库存分析】设计储位类型维度计算出错！！！')

    def inventory_current_location(self, index=None, pt_col=None):
        try:
            # print('In current_location function')

            # inv_dateCol = self.inv_df.columns[int(self.inv_dateCol.text())]  # 库存日期
            # inv_putawayDateCol = self.inv_df.columns[int(self.inv_putawayDateCol.text())]  # 上架日期
            # inv_skuCol = self.inv_df.columns[int(self.inv_skuCol.text())]  # 产品代码
            # inv_invQty = self.inv_df.columns[int(self.inv_invQty.text())]  # 在库件数
            # inv_invVol = self.inv_df.columns[int(self.inv_invVol.text())]  # 在库体积
            # inv_invTypeCol = self.inv_df.columns[int(self.inv_locTypeCol.text())]  # 储位类型
            # inv_locNumCol = self.inv_df.columns[int(self.inv_locNumCol.text())]  # 在库数量

            # 库存数据中的列名
            [inv_dateCol, inv_putawayDateCol, inv_customerCol, inv_skuCol, inv_skuSizeCol, inv_invQty, inv_invVol,
             inv_invTypeCol, inv_skuLengthCol, inv_skuWidthCol, inv_skuHeightCol, inv_locNumCol] = self.inv_df.columns[0:12]

            if index is None:
                index = [inv_dateCol, inv_invTypeCol]

            if pt_col is None:
                pt_col = [inv_skuCol, inv_locNumCol, inv_invQty, inv_invVol]

            # datetime64[ns]不能作为key,将日期列的格式转换为string
            self.inv_df[inv_dateCol] = self.inv_df[inv_dateCol].astype(str)
            self.inv_df[inv_putawayDateCol] = self.inv_df[inv_putawayDateCol].astype(str)

            # datetime64[ns]不能作为key,将日期列的格式转换为string
            self.inv_df[inv_dateCol] = self.inv_df[inv_dateCol].astype(str)
            # print('库存数据： \n', self.inv_df.head(10))

            df_location = pd.pivot_table(self.inv_df, index=index,
                                         values=pt_col,
                                         aggfunc={inv_skuCol: pd.Series.nunique, inv_locNumCol: np.sum, inv_invQty: np.sum, inv_invVol: np.sum},
                                         fill_value=0).reset_index()

            # 重排列
            re_pt_col = [inv_invVol, inv_invQty, '产品代码', '库位数量']
            df_location = df_location[index + re_pt_col]

            # 重命名列
            df_location.columns = index + ['在库体积(m³)', '在库件数', 'sku数', '库位数量']

            # 透视字段， 需计算总和及百分比的字段
            sum_col = ['在库体积(m³)', '在库件数', 'sku数', '库位数量']

            row_n = df_location.shape[0]
            df_location.at[row_n, index[-1:]] = '合计'
            df_location.at[row_n, sum_col] = df_location[sum_col].apply(lambda x: x.sum())
            # 计算比例
            for i in range(len(sum_col)):
                df_location[sum_col[i] + '%'] = df_location[sum_col[i]] / (df_location[sum_col[i]].sum() / 2)

            # 计算库存深度
            df_location['库存深度(m³/sku)'] = df_location['在库体积(m³)'] / df_location['sku数']
            df_location['库存深度(件/sku)'] = df_location['在库件数'] / df_location['sku数']
            return df_location

        except:
            self.show_error_dialog('【库存分析】储位类型维度计算出错！！！')

    def inventory_customer_pivot(self):
        '''
        计算客户维度的件型分布
        '''
        try:
            # 输入库存数据对应列编号
            # inv_customerCol = self.inv_df.columns[int(self.inv_customerCol.text())]    # 客户代码
            # inv_skuCol = self.inv_df.columns[int(self.inv_skuCol.text())]  # 产品代码
            # inv_skuSizeCol = self.inv_df.columns[int(self.inv_skuSizeCol.text())]  # 产品货型
            # inv_invQty = self.inv_df.columns[int(self.inv_invQty.text())]    # 在库件数
            # inv_invVol = self.inv_df.columns[int(self.inv_invVol.text())]    # 在库体积

            # 库存数据中的列名
            [inv_dateCol, inv_putawayDateCol, inv_customerCol, inv_skuCol, inv_skuSizeCol, inv_invQty, inv_invVol,
             inv_invTypeCol, inv_skuLengthCol, inv_skuWidthCol, inv_skuHeightCol, inv_locNumCol] = self.inv_df.columns[0:12]

            sort_size = ['XL', 'L2', 'L1', 'M', 'S', 'XS']
            self.inv_df[inv_skuSizeCol] = pd.Categorical(self.inv_df[inv_skuSizeCol], sort_size)

            ## 客户在库体积及在库件数
            df_customer = pd.pivot_table(self.inv_df, index=[inv_customerCol],
                                         values=[inv_invVol, inv_invQty],
                                         columns=[inv_skuSizeCol],
                                         aggfunc='sum',
                                         margins=True,
                                         margins_name='总计',
                                         fill_value=0).reset_index()

            ## 按客户总体积排序
            df_customer = df_customer.sort_values(by=(inv_invVol, '总计'), ascending=False, ignore_index=True)

            ### 多级索引转成单层索引
            col = []
            for (s1, s2) in df_customer.columns:
                if len(s2) > 0:
                    col.append(s1 + '_' + str(s2))
                else:
                    col.append(s1)
            # delivery_df.columns = [ s1 + '_' + str(s2) for (s1, s2) in delivery_df.columns]
            df_customer.columns = col

            ## 计算体积货型占比
            for item in sort_size:
                df_customer[('{}_{}%'.format(inv_invVol, item))] = df_customer[('{}_{}'.format(inv_invVol, item))] / df_customer[('{}_总计'.format(inv_invVol))]

            ## 计算件数货型占比
            for item in sort_size:
                df_customer[('{}_{}%'.format(inv_invQty, item))] = df_customer[('{}_{}'.format(inv_invQty, item))] / df_customer[('{}_总计'.format(inv_invQty))]

            ## 客户sku数
            # sku非重复计数
            customer_sku = self.inv_df[[inv_customerCol, inv_skuCol]].groupby(inv_customerCol).nunique()
            # print('1 客户sku统计： ', customer_sku)
            customer_sku = (pd.DataFrame(customer_sku)).reset_index()
            # print('2 客户sku统计： ', customer_sku)
            customer_sku.columns = ['客户代码', 'sku数']
            total_sku = sum(customer_sku['sku数'])

            ### 合并库存货型分布及sku数
            df_customer = pd.merge(df_customer, customer_sku, how='left', sort=False)

            ## 修改sku的总计值
            df_customer.loc[(df_customer[inv_customerCol] == '总计'), ['sku数']] = total_sku

            ### 计算库存深度
            df_customer['库存深度(m³/sku)'] = df_customer['{}_总计'.format(inv_invVol)] / df_customer['sku数']
            df_customer['库存深度(件/sku)'] = df_customer['{}_总计'.format(inv_invQty)] / df_customer['sku数']

            df_customer['大件体积占比'] = df_customer['{}_XL%'.format(inv_invVol)] + df_customer['{}_L2%'.format(inv_invVol)] + df_customer['{}_L1%'.format(inv_invVol)]

            df_customer['客户类型'] = np.NAN
            df_customer.loc[(df_customer['客户类型'].isna()) & (df_customer['大件体积占比'] >= 0.8), ['客户类型']] = '纯大货型'
            df_customer.loc[(df_customer['客户类型'].isna()) & (df_customer['大件体积占比'] >= 0.6), ['客户类型']] = '大货型'
            df_customer.loc[(df_customer['客户类型'].isna()) & (df_customer['大件体积占比'] >= 0.3), ['客户类型']] = '混货型'
            df_customer.loc[(df_customer['客户类型'].isna()), ['客户类型']] = '小货型'

            ### 重排列
            org_customer_columns = [inv_customerCol, '{}_XL'.format(inv_invVol), '{}_L2'.format(inv_invVol), '{}_L1'.format(inv_invVol),
                                    '{}_M'.format(inv_invVol), '{}_S'.format(inv_invVol), '{}_XS'.format(inv_invVol), '{}_总计'.format(inv_invVol),
                                    '{}_XL%'.format(inv_invVol), '{}_L2%'.format(inv_invVol), '{}_L1%'.format(inv_invVol),
                                    '{}_M%'.format(inv_invVol), '{}_S%'.format(inv_invVol), '{}_XS%'.format(inv_invVol),

                                    '{}_XL'.format(inv_invQty), '{}_L2'.format(inv_invQty), '{}_L1'.format(inv_invQty),
                                    '{}_M'.format(inv_invQty), '{}_S'.format(inv_invQty), '{}_XS'.format(inv_invQty), '{}_总计'.format(inv_invQty),
                                    '{}_XL%'.format(inv_invQty), '{}_L2%'.format(inv_invQty), '{}_L1%'.format(inv_invQty),
                                    '{}_M%'.format(inv_invQty), '{}_S%'.format(inv_invQty), '{}_XS%'.format(inv_invQty),

                                    'sku数', '库存深度(m³/sku)', '库存深度(件/sku)', '大件体积占比', '客户类型']

            df_customer = df_customer[org_customer_columns]
            return df_customer
        except:
            self.show_error_dialog('【库存分析】客户维度计算出错！！！！！！')

    def inventory_get_sku_pivot(self, skuSize, invQty, invVol):
        '''
        :param df: 数据源
        :return: 返回sku维度的在库件数、在库体积、库龄的透视表
        '''
        try:
            ## sku在库体积及在库件数
            df_vol = pd.pivot_table(self.inv_df, index=[skuSize],
                                    values=[invQty, invVol],
                                    aggfunc=np.sum,
                                    margins=False,
                                    fill_value=0).reset_index()

            ## sku在库件数的加权库龄
            df_age = pd.pivot_table(self.inv_df, index=[skuSize],
                                    values=['库龄'],
                                    aggfunc=lambda rows: np.average(rows, weights=self.inv_df.loc[rows.index, invQty]),
                                    margins=False,
                                    fill_value=0).reset_index()

            df_sku = pd.merge(df_vol, df_age, on=[skuSize])

            ## 计算sku维度库龄类别   AGE_CLASS 数据结构为[['D1(0,30]', 0, 30], ['D2(30,60]', 30, 60]] 二维列表，其中的元素为【等级，左区间，右区间】
            df_sku['库龄等级'] = ''
            rank_num = len(config.AGE_CLASS)
            for i in range(rank_num):
                df_sku.loc[(df_sku['库龄'] > config.AGE_CLASS[i][1]) & (df_sku['库龄'] <= config.AGE_CLASS[i][2]), ['库龄等级']] = config.AGE_CLASS[i][0]

            sku_col = ['sku', 'inv_quantity', 'inv_Vol(m³)', 'age', 'age_class']
            df_sku.columns = sku_col

            return df_sku
        except:
            self.show_error_dialog('【库存分析】SKU维度计算出错！！！')

    def inventory_get_customer_pivot(self):
        ### 客户维度的库存，sku数，库存深度
        # 输入库存数据对应列编号
        try:
            # inv_customerCol = self.inv_df.columns[int(self.inv_customerCol.text())]  # 客户代码
            # inv_skuCol = self.inv_df.columns[int(self.inv_skuCol.text())]  # 产品代码
            # inv_invQty = self.inv_df.columns[int(self.inv_invQty.text())]  # 在库件数
            # inv_invVol = self.inv_df.columns[int(self.inv_invVol.text())]  # 在库体积

            # 库存数据中的列名
            [inv_dateCol, inv_putawayDateCol, inv_customerCol, inv_skuCol, inv_skuSizeCol, inv_invQty, inv_invVol,
             inv_invTypeCol, inv_skuLengthCol, inv_skuWidthCol, inv_skuHeightCol, inv_locNumCol] = self.inv_df.columns[0:12]

            customer_cols = [inv_customerCol, inv_skuCol, inv_invQty, inv_invVol]

            customer_df = pd.pivot_table(self.inv_df[customer_cols], index=[inv_customerCol],
                                         values=[inv_skuCol, inv_invQty, inv_invVol],
                                         aggfunc={inv_skuCol:pd.Series.nunique, inv_invQty:np.sum, inv_invVol:np.sum},
                                         margins=True,
                                         margins_name='总计',
                                         fill_value=0).reset_index()

            customer_df.columns = ['customer', 'sku数', '在库件数', '在库体积(m³)']
            customer_df['库存深度(件/sku)'] = customer_df['在库件数']/customer_df['sku数']
            customer_df['库存深度(m³/sku)'] = customer_df['在库体积(m³)'] / customer_df['sku数']
            return customer_df
        except:
            self.show_error_dialog('库存数据-客户维度分析错误！')

    def inventory_save_data(self):
        try:
            filePath, ok2 = QFileDialog.getSaveFileName(None, caption='保存文件', filter='Excel files(*.xlsx , *.xls)')
            print(filePath)  # 打印保存文件的全部路径（包括文件名和后缀名）
            if 'xls' in filePath or 'xlsx' in filePath:
                ### write to file
                writer = pd.ExcelWriter(filePath)

                print('inventory data info: ', self.inv_info)
                if self.design_location_df.shape[0]>0:
                    print('self.design_location_df: ', self.design_location_df)
                    self.format_data(writer=writer, df=self.design_location_df, sheet_name='A1.1-库位推荐', source_data_info=self.inv_info)
                if self.current_location_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.current_location_df, sheet_name='A1.2-现状库位', source_data_info=self.inv_info)
                if self.inventory_customer_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.inventory_customer_df, sheet_name='A2-客户货型分布', source_data_info=self.inv_info)
                if self.inventory_age_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.inventory_age_df, sheet_name='A3-库龄', source_data_info=self.inv_info)
                if self.inv_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.inv_df, sheet_name='01-数据源')
                if self.sku_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.sku_df, sheet_name='02-sku维度数据源')

                writer.save()
                self.show_info_dialog('库存分析 结果保存成功！')

            else:
                self.show_info_dialog('请保存为指定的文件类型！')
        except:
            self.show_error_dialog('文件保存失败！')


    def inventory_clear_data(self):
        try:
            self.inv_totalRow.clear()  # 数据总行数
            self.inv_totalCol.clear()  # 数据总列数

            self.inv_errorRow.clear()  # 重复行数
            self.inv_validRow.clear()  # 数据去重后总行数

            self.inv_totalSKU.clear()
            self.inv_totalDays.clear()

            print('数据清理完成！')

            print('清理库存数据！')

            self.inv_df = pd.DataFrame()
            self.sku_df = pd.DataFrame()


            # self.inv_tableView.reset()
            self.inv_tableView.setModel(dfModel(pd.DataFrame()))
            self.sku_tableView.setModel(dfModel(pd.DataFrame()))
            print('清理SKU数据！')

            # print('库存数据： ', self.inv_df)
            # print('SKU数据： ', self.sku_df)
        except:
            self.show_error_dialog('数据重置失败！！！')



    def outbound_load_order_data(self):
        try:
            filenames = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls, *.csv)')
            filename = filenames[0]

            if 'csv' in filename:
                try:
                    self.ob_df = pd.read_csv(filename, encoding='utf-8')
                except:
                    self.ob_df = pd.read_csv(filename, encoding='gbk')
                print('出库数据1： ', self.ob_df.shape)
                row1 = self.ob_df.shape[0]

                # 删除有空值的行
                self.ob_df.dropna(how='all', inplace=True)
                print('出库数据2： ', self.ob_df.shape)
                row2 = self.ob_df.shape[0]
                model = dfModel(self.ob_df.head(100))
                self.ob_totalRow.setText("{:,}".format(row2))  # 数据总行数
                self.ob_errorRow.setText("{:,}".format(row1 - row2))  # 异常数据
                self.ob_tableView.setModel(model)
            elif 'xlsx' in filename:
                self.ob_df = pd.read_excel(filename)
                # 删除有空值的行
                row1 = self.ob_df.shape[0]
                self.ob_df.dropna(how='all', inplace=True)
                row2 = self.ob_df.shape[0]
                self.ob_totalRow.setText("{:,}".format(row2))  # 数据总行数
                self.ob_errorRow.setText("{:,}".format(row1 - row2))  # 异常数据
                model = dfModel(self.ob_df.head(100))
                self.ob_tableView.setModel(model)
            else:
                self.show_error_dialog('请选择csv或xlsx文件类型!')
        except:
            self.show_error_dialog('请选择文件!')


    def outbound_load_order_time_data(self):
        try:
            filenames = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls, *.csv)')
            filename = filenames[0]

            if 'csv' in filename:
                try:
                    self.ob_time_df = pd.read_csv(filename, encoding='utf-8')
                except:
                    self.ob_time_df = pd.read_csv(filename, encoding='gbk')

                # 删除有空值的行
                self.ob_time_df.dropna(how='all', inplace=True)
                model = dfModel(self.ob_time_df.head(100))
                self.ob_timeTableView.setModel(model)

            elif 'xlsx' in filename:
                self.ob_time_df = pd.read_excel(filename)
                # 删除有空值的行
                self.ob_time_df.dropna(how='all', inplace=True)
                model = dfModel(self.ob_time_df.head(100))
                self.ob_timeTableView.setModel(model)
            else:
                self.show_error_dialog('请选择csv或xlsx文件类型!')
        except:
            self.show_error_dialog('请选择文件!')


    def outbound_data_process(self):
        try:
            '''出库数据'''
            print('in outbound_data_process function ~~~~~~~~~~')
            date = self.ob_df.columns[int(self.ob_dateCol.text())]       # 出库日期
            orderID = self.ob_df.columns[int(self.ob_orderCol.text())]   # 订单号
            sku = self.ob_df.columns[int(self.ob_skuCol.text())]         # sku
            quantity = self.ob_df.columns[int(self.ob_qtyCol.text())]    # 出库件数

            option_orig_column = []
            option_new_column = []


            if self.ob_orderTagCol.text() != "":
                order_tag = self.ob_df.columns[int(self.ob_orderTagCol.text())]  # 订单标识
                option_orig_column.append(order_tag)
                option_new_column.append('order_tag')
            if self.ob_skuSizeCol.text() != "":
                sku_size = self.ob_df.columns[int(self.ob_skuSizeCol.text())]  # 产品货型
                option_orig_column.append(sku_size)
                option_new_column.append('sku_size')
            if self.ob_putawayDateCol.text() != "":
                putaway_date = self.ob_df.columns[int(self.ob_putawayDateCol.text())]  # 上架时间
                option_orig_column.append(putaway_date)
                option_new_column.append('putaway_date')
            if self.ob_pickupNoCol.text() != "":
                pickupNO = self.ob_df.columns[int(self.ob_pickupNoCol.text())]  # 拣货单号
                option_orig_column.append(pickupNO)
                option_new_column.append('pickupNO')
            if self.ob_locationCol.text() != "":
                location = self.ob_df.columns[int(self.ob_locationCol.text())]  # 库位代码
                option_orig_column.append(location)
                option_new_column.append('location')

            print('option_orig_column: ', option_orig_column)

            valid_columns_name = [date, orderID, sku, quantity]+ option_orig_column
            new_columns_name = ['date', 'orderID', 'sku', 'quantity'] + option_new_column

            print('出库数据有效列名：', valid_columns_name)
            detail_data = self.ob_df[valid_columns_name]
            detail_data.columns = new_columns_name

            ### 删除有缺失的列
            # self.ob_df.dropna(how='any', inplace=True)
            detail_data.dropna(how='any', inplace=True)

            print(detail_data.head(10))
            # row1 = self.ob_df.shape[0]

            '''出库时间列 数据处理'''
            date_col = []
            for x in new_columns_name:
                if 'date' in x:
                    print('时间列： ', x)
                    date_col.append(x)
                    ### 填充上架日期的缺失值
                    detail_data[x] = detail_data[x].fillna(datetime.datetime.now().strftime('%Y年%m月%d日'))
                    detail_data.loc[(detail_data[x] == '*'), [x]] = datetime.datetime.now().strftime('%Y年%m月%d日')

            for col in date_col:
                print('修改时间列时间格式： ', col)
                try:
                    detail_data[col] = pd.to_datetime(detail_data[col], format='%Y/%m/%d')
                except ValueError:
                    detail_data[col] = pd.to_datetime(detail_data[col], format='%Y年%m月%d日')
                else:
                    pass
            ## 新增 客户代码列
            detail_data['customer'] = detail_data['orderID'].map(lambda x: x[0: x.index('-')])

            print('订单数据处理完成！')
            print(detail_data.head(10))

            # self.show_info_dialog('订单明细数据处理完成！！')

            '''
            订单流入数据
            '''
            orderID_obtime = self.ob_time_df.columns[int(self.ob_orderCol2.text())]  # 订单号
            time_in = self.ob_time_df.columns[int(self.ob_timeInCol.text())]  # sku

            print('orderID_obtime , time_in: ', orderID_obtime, time_in)

            option_orig_column2 = []
            option_new_column2 = []

            if self.ob_orderStructureCol.text() != "":
                order_structure = self.ob_time_df.columns[int(self.ob_orderStructureCol.text())]  # 订单结构
                option_orig_column2.append(order_structure)
                option_new_column2.append('order_structure')
            if self.ob_platformCol.text() != "":
                platform = self.ob_time_df.columns[int(self.ob_platformCol.text())]  # 平台
                option_orig_column2.append(platform)
                option_new_column2.append('platform')
            if self.ob_channelCol.text() != "":
                channel = self.ob_time_df.columns[int(self.ob_channelCol.text())]  # 物流渠道
                option_orig_column2.append(channel)
                option_new_column2.append('channel')

            print('option_orig_column2：', option_orig_column2)
            print('option_new_column2：', option_new_column2)


            valid_columns_name2 = [orderID_obtime, time_in] + option_orig_column2
            new_columns_name2 = ['orderID', 'time_in'] + option_new_column2


            print('订单流入时间有效列名：', valid_columns_name2)
            time_data = self.ob_time_df[valid_columns_name2]

            ### 筛查有缺失的列
            self.ob_time_df.dropna(how='any', inplace=True)

            time_data.columns = new_columns_name2
            time_data['time_in'] = pd.to_datetime(time_data['time_in'])

            # 合并出库数据和订单流入时间数据
            re = pd.merge(detail_data, time_data, on='orderID', how='left')
            print('合并数据列名： ', re.columns)
            ### 增加出库库龄
            re['sku_age'] = pd.to_timedelta(re['date'] - re['putaway_date']).dt.days
            re.loc[(re['sku_age'] < 0), ['sku_age']] = 30  ### 库龄异常的行，赋值为30天

            print('合并数据： ', re.shape)
            print('合并数据： ', re.shape)
            print(re)

            re.dropna(how='any', inplace=True)  # 删除有空值的行
            self.ob_validDF = re
            print('self.ob_validDF shape: ', self.ob_validDF.shape)
            print(self.ob_validDF.head(10))
            print(self.ob_validDF.columns)
            print('合并订单数据及订单流入时间 成功！')

            # self.show_info_dialog('订单流入时间数据处理完成！！')

            self.ob_tableView.setModel(dfModel(self.ob_validDF.head(100)))
            self.ob_timeTableView.setModel(dfModel(pd.DataFrame()))


            '''数据预览'''

            self.ob_validRow.setText("{:,}".format(self.ob_validDF.shape[0]))  # 有效数据总行数

            self.ob_totalOrders.setText("{:,}".format(self.ob_validDF['orderID'].nunique()))  # 总订单数
            self.ob_totalSKU.setText("{:,}".format(self.ob_validDF['sku'].nunique()))     # 总SKU数
            self.ob_totalDays.setText("{:,}".format(self.ob_validDF['date'].nunique()))   # 总出库天数



            '''
            ===========================================================================
            订单数据处理
            将订单行明细数据处理成订单维度的 中间数据
            ===========================================================================
            '''

            ## GC数据 更新订单类型
            self.ob_validDF.loc[(self.ob_validDF['order_tag'] == '是'), ['order_tag']] = 'FBA订单'
            self.ob_validDF.loc[(self.ob_validDF['order_tag'] == '否'), ['order_tag']] = '标准订单'

            ### outbound source information
            print('***************')
            df_order = pd.pivot_table(self.ob_validDF, index=['orderID'], values=['sku', 'quantity'], aggfunc={'sku': len, 'quantity': np.sum}).reset_index()
            ## 根据订单的行数和件数更新订单结构
            df_order['re_order_structure'] = np.NAN
            df_order.loc[(df_order['sku'] == 1) & (df_order['quantity'] == 1), ['re_order_structure']] = '单品单件'
            df_order.loc[(df_order['sku'] == 1) & (df_order['quantity'] > 1), ['re_order_structure']] = '单品多件'
            df_order.loc[(df_order['sku'] >= 10) | (df_order['quantity'] >= 20), ['re_order_structure']] = '批量订单'
            df_order.loc[(df_order['sku'] > 1), ['re_order_structure']] = '多品多件'

            ### 增加新定义的订单类型
            self.ob_validDF = pd.merge(self.ob_validDF, df_order[['orderID', 're_order_structure']], on=['orderID'], how='left')
            print('0000000000', self.ob_validDF.shape)


            ''' 计算订单的货型组合 '''
            temp_df_order_size = self.ob_validDF[['orderID', 'sku_size']]

            dict = {'XS': '小', 'S': '小', 'M': '中', 'L1': '大', 'L2': '大', 'XL': 'XL'}

            temp_df_order_size['size_type'] = temp_df_order_size['sku_size'].copy().map(dict)
            sort_size = ['小', '中', '大', 'XL']
            temp_df_order_size['size_type'] = pd.Categorical(temp_df_order_size['size_type'], sort_size)

            df_order_size = temp_df_order_size[['orderID', 'size_type']].groupby('orderID')['size_type'].agg('-'.join).reset_index()

            ### 增加订单货型字段
            df_order_size['order_size_type'] = df_order_size['size_type']
            print('11111111, df_order_size', df_order_size.head(10))

            ###组合类型
            df_order_size.loc[(df_order_size['size_type'].str.contains('大')) & (df_order_size['size_type'].str.contains('小')), 'order_size_type'] = '大配小'
            df_order_size.loc[(df_order_size['size_type'].str.contains('大')) & (df_order_size['size_type'].str.contains('中')), 'order_size_type'] = '大配中'
            df_order_size.loc[(df_order_size['size_type'].str.contains('中')) & (df_order_size['size_type'].str.contains('小')), 'order_size_type'] = '中配小'
            df_order_size.loc[(df_order_size['size_type'].str.contains('大')) & (df_order_size['size_type'].str.contains('中'))
                              & (df_order_size['size_type'].str.contains('小')), 'order_size_type'] = '大中小'

            ### 单类型
            df_order_size.loc[
                (df_order_size['size_type'].str.contains('大')) & ~(df_order_size['size_type'].str.contains('中')) & ~(
                    df_order_size['size_type'].str.contains('小')), 'order_size_type'] = '大'
            df_order_size.loc[
                (df_order_size['size_type'].str.contains('中')) & ~(df_order_size['size_type'].str.contains('大')) & ~(
                    df_order_size['size_type'].str.contains('小')), 'order_size_type'] = '中'
            df_order_size.loc[
                (df_order_size['size_type'].str.contains('小')) & ~(df_order_size['size_type'].str.contains('中')) & ~(
                    df_order_size['size_type'].str.contains('大')), 'order_size_type'] = '小'

            df_order_size.loc[(df_order_size['size_type'].str.contains('XL')), 'order_size_type'] = 'XL'

            # order_size 按指定顺序排列
            order_type_sorted = ['小', '中', '大', '中配小', '大配小', '大配中', '大中小', 'XL']
            df_order_size['order_size_type'] = pd.Categorical(df_order_size['order_size_type'], order_type_sorted)

            print('22222222, df_order_size', df_order_size.head(10))

            '''合并 订单明细数据 和 订单货型数据'''
            self.ob_validDF = pd.merge(self.ob_validDF, df_order_size, on=['orderID'], how='left')

            print('33333333, ob_df', self.ob_df.columns)
            # print('33333333, ob_df', self.ob_df.dtypes)

            self.ob_validDF['month'] = self.ob_validDF.date.dt.month
            self.ob_validDF['week'] = self.ob_validDF.date.dt.isocalendar().week
            self.ob_validDF['weekday'] = self.ob_validDF.date.dt.weekday + 1  # 内置0~6的序列表示星期一到星期日，为引起歧义修改为1~7
            self.ob_validDF['hour'] = self.ob_validDF.time_in.dt.hour

            cutoff_hour = int(self.ob_hourinCutoffHour.text())

            self.ob_validDF['wave'] = 'Wave1'
            self.ob_validDF.loc[(self.ob_validDF['hour'] >= 7) & (self.ob_validDF['hour'] <= cutoff_hour), ['wave']] = 'Wave2'
            self.show_info_dialog('出库【数据处理】完成！')

        except:
            self.show_error_dialog('请输入必填字段对应的列编号！')


    def outbound_EIQ_analysis(self):
        try:

            '''
            订单EIQ分析，输入参数
            '''
            normal_order = '标准订单'
            fba_order = 'FBA订单'

            '''标准订单EIQ'''
            order_index = ['order_tag']
            df_normal = self.ob_validDF.query('order_tag == "{}"'.format(normal_order))
            self.normal_order_EIQ = self.get_EIQ(df=df_normal, index=order_index)

            '''FBA订单EIQ'''
            df_fba = self.ob_validDF.query('order_tag == "{}"'.format(fba_order))
            self.fba_order_EIQ = self.get_EIQ(df=df_fba, index=order_index)

            '''订单结构EIQ'''
            order_index = ['order_tag', 're_order_structure']
            self.order_type_EIQ = self.get_EIQ(df=df_normal, index=order_index, isPercentage=True)

            # print('\n')
            # print('*'*10, '订单结构维度的EIQ', '*'*10)
            # print(order_type_EIQ)

            '''波次EIQ'''
            wave_index = ['wave', 're_order_structure']
            wave_date = self.inv_dateEdit.text()
            df_wave = self.ob_validDF.query('date >= "{}"'.format(wave_date))
            df_wave['date'] = df_wave['date'].astype(np.str)  # datetime64[ns] can't be the merge index
            self.wave_order_type_EIQ = self.get_EIQ(df=df_wave, index=wave_index)
            # print('\n')
            # print('*' * 10, 'wave_order_type_EIQ', '*' * 10)
            # print(wave_order_type_EIQ)

            '''现状拣选行EIQ'''
            current_index = ['order_tag']
            self.current_pick_order_EIQ = self.get_pick_EIQ(df=df_normal, index=current_index)

            print('显示标准订单EIQ~~~~~~')
            self.ob_timeTableView.setModel(dfModel(self.normal_order_EIQ))
            self.show_info_dialog('EIQ分析计算完成！')

        except:
            self.show_error_dialog('EIQ计算错误！！！请确认字段是否正确！！！')


    def outbound_multi_order_analysis(self):
        try:
            print('In outbound_multi_order_analysis function!!!')

            '''多品多件订单货型组合'''
            df_multi_order = self.ob_validDF.query('re_order_structure == "{}"'.format('多品多件'))
            multi_index = ['order_tag', 're_order_structure', 'order_size_type']

            self.multi_order_EIQ = self.order_distribution(df=df_multi_order, index=multi_index)
            self.ob_timeTableView.setModel(dfModel(self.multi_order_EIQ))
            self.show_info_dialog('【多品订单组合分析】计算完成！')

        except:
            self.show_error_dialog('【多品订单组合分析】计算错误！！！请确认字段是否正确！！！')


    def outbound_customer_sale_analysis(self):
        try:
            print('In outbound_customer_sale_analysis function!!!')
            customer_index = ['customer']
            customer_qty = pd.pivot_table(self.ob_validDF, index=customer_index,
                                          values=['quantity', 'date'],
                                          columns=['re_order_structure'],
                                          aggfunc={'quantity': np.sum, 'date': pd.Series.nunique},
                                          margins=True,
                                          margins_name='总计',
                                          fill_value=0).reset_index()
            ### 多级索引转成单层索引
            cols = []
            for (s1, s2) in customer_qty.columns:
                if len(s2) > 0:
                    cols.append(s1 + '_' + str(s2))
                else:
                    cols.append(s1)
            customer_qty.columns = cols

            order_type = ['单品单件', '单品多件', '多品多件', '批量订单', '总计']
            # 计算日均出库件数
            for i in range(len(order_type)):
                customer_qty['日均件_' + order_type[i]] = 0
                customer_qty.loc[(customer_qty['date_' + order_type[i]]) > 0, ['日均件_' + order_type[i]]] = customer_qty['quantity_' + order_type[i]] / customer_qty[
                    'date_' + order_type[i]]

            col = ['customer', 'quantity_总计', 'date_总计', '日均件_总计', '日均件_单品单件', '日均件_单品多件', '日均件_多品多件', '日均件_批量订单']
            re_col = ['customer', '出库总件数', '出库天数', '日均出库件数', '日均件_单品单件', '日均件_单品多件', '日均件_多品多件', '日均件_批量订单']

            customer_qty = customer_qty[col]
            customer_qty.columns = re_col

            '''月&日均动销SKU'''
            month_cnt = pd.pivot_table(self.ob_validDF, index=['month'],
                                       values=['date'],
                                       aggfunc=pd.Series.nunique).reset_index()

            # 查询最大的月份
            select_month = 0
            days = 0
            # 选取天数最大的月份
            for index, row in month_cnt.iterrows():
                if row['date'] > days:
                    days = row['date']
                    select_month = row['month']
            # 数据源中总天数小于等于30天，月动销sku不做筛选
            if month_cnt['date'].sum() <= 30:
                select_month = 0

            if select_month > 0:
                active_df = self.ob_validDF.query('month == {}'.format(select_month))
            else:
                active_df = self.ob_validDF

            customer_month_sku = pd.pivot_table(active_df, index=customer_index,
                                                values=['sku'],
                                                aggfunc=pd.Series.nunique,
                                                margins=True,
                                                margins_name='总计',
                                                fill_value=0).reset_index()
            customer_month_sku.columns = customer_index + ['月动销sku']

            print('active_df: ', active_df.columns)
            print('active_df shape: ', active_df.shape)

            sku_index = ['customer', 'date']
            customer_daily_sku2 = active_df[sku_index + ['sku']].groupby(sku_index).nunique().reset_index()

            customer_daily_sku = pd.pivot_table(customer_daily_sku2, index=customer_index,
                                                values=['sku'],
                                                aggfunc=pd.Series.nunique,
                                                margins=True,
                                                margins_name='总计',
                                                fill_value=0).reset_index()

            print('customer_daily_sku: ', customer_daily_sku.columns)
            print('customer_daily_sku: ', customer_daily_sku.head(5))

            customer_daily_sku.columns = customer_index + ['日均动销sku']
            # customer_daily_sku['日均动销sku'] = customer_daily_sku[customer_daily_sku>0].mean(axis=1)

            customer_sku = pd.merge(customer_month_sku, customer_daily_sku, on='customer', how='left')
            # 计算库存数据中客户维度在库存数据
            inv_customer_df = self.inventory_get_customer_pivot()
            # 合并库存 和 出库的客户数据
            customer_temp = pd.merge(inv_customer_df, customer_qty, on='customer', how='left')

            customer_re = pd.merge(customer_temp, customer_sku, on='customer', how='left').fillna(0)
            # print('customer_re detail: \n')
            # print(customer_re.head(10))

            '''增加整体计算值'''
            customer_re['sku月动销率'] = customer_re['月动销sku'] / customer_re['sku数']
            customer_re['sku日动销率'] = customer_re['日均动销sku'] / customer_re['sku数']
            # customer_re.loc[(customer_re['sku月动销率'] >= 1), ['sku月动销率']] = 1

            customer_re['库存周期'] = customer_re['在库件数'] / customer_re['日均出库件数']
            customer_re = customer_re.sort_values(by='日均出库件数', ascending=False, ignore_index=True)

            self.customer_sales = customer_re
            self.ob_timeTableView.setModel(dfModel(self.customer_sales))
            self.show_info_dialog('【客户动销】计算完成！')
        except:
            self.show_error_dialog('【客户动销】分析计算错误！请确认出库数据及字段名是否输入正确！！！')



    def outbound_sku_active_sale_analysis(self):
        try:
            print('In outbound_sku_sale_analysis function!!!')

            active_col = ['week', 'sku', 'quantity']
            df = self.ob_validDF[active_col]

            print('000000 df:', df.head(10))

            index = ['week']
            # df = df.sort_values(by=index).reset_index()  # 按周排序
            # print('1111111 df:', df.head(10))
            groups_df = df.groupby(index)
            print('testing 1111 groups_df: ', groups_df)

            group_dict = {}
            sku_dict = {}
            for item, group in groups_df:
                sku_dict[item] = set(group['sku'])
                group_dict[item] = group

            sku_cnt = []
            for k, v in sku_dict.items():
                curr = k
                if k + 1 > 52:
                    next = 1
                else:
                    next = k + 1

                curr_cnt = len(v)
                next_cnt = len(sku_dict.get(next, {}))

                ### sku交集
                comm_sku_set = v.intersection(sku_dict.get(next, {}))
                curr_df = group_dict[curr]
                curr_qty = curr_df['quantity'].sum()
                curr_comm_qty = curr_df[curr_df['sku'].isin(list(comm_sku_set))]['quantity'].sum()

                next_df = group_dict.get(next, None)
                if next_df is None:
                    next_qty = 0
                    next_comm_qty = 0
                else:
                    next_qty = next_df['quantity'].sum()
                    next_comm_qty = next_df[next_df['sku'].isin(list(comm_sku_set))]['quantity'].sum()

                comm_cnt = len(v.intersection(sku_dict.get(next, {})))
                in_cnt = len(set(sku_dict.get(next, {})).difference(v))
                out_cnt = len(v.difference(sku_dict.get(next, {})))

                sku_cnt.append([curr, next, curr_cnt, next_cnt, comm_cnt, in_cnt, out_cnt, curr_qty, next_qty, curr_comm_qty, next_comm_qty])

            sku_col = ['curr_{}'.format(index[-1]), 'next_{}'.format(index[-1]), 'curr_sku', 'next_sku', '重合sku', '流入sku', '流出sku',
                       'current件数', 'next件数', 'current重合sku件数', 'next重合sku件数']

            print('testing 2222 sku_cnt: ', sku_cnt)

            sku_df = pd.DataFrame(sku_cnt, columns=sku_col)
            print('sku动销的df, sku_df 111:', sku_df)

            sku_df['sku池变化率'] = sku_df['流入sku'] / sku_df['curr_sku']
            sku_df['current重合sku件数%'] = sku_df['current重合sku件数'] / sku_df['current件数']
            sku_df['next重合sku件数%'] = sku_df['next重合sku件数'] / sku_df['next件数']

            sku_df = sku_df.sort_values(by=['next_{}'.format(index[-1])]).fillna(0)

            print('sku动销的df, sku_df 222:', sku_df)
            self.sku_active_sales = sku_df
            self.ob_timeTableView.setModel(dfModel(self.sku_active_sales))
            self.show_info_dialog('【SKU动销】计算完成！')
        except:
            self.show_error_dialog('【SKU动销】分析计算错误！！！请确认出库数据及字段名是否输入正确！！！')

    def outbound_sku_age_analysis(self):
        try:
            print('In outbound_sku_age_analysis function!!!')
            # 实例化配置参数
            config = Config()
            config.run()

            outbound_age_col = ['orderID', 'sku', 'quantity', 'customer', 're_order_structure', 'sku_age']
            df = self.ob_validDF[outbound_age_col]

            ''' 计算sku出库库龄等级'''
            df['sku_age_class'] = ''
            rank_num = len(config.AGE_CLASS)
            for i in range(rank_num):
                df.loc[(df['sku_age'] > config.AGE_CLASS[i][1]) & (df['sku_age'] <= config.AGE_CLASS[i][2]), ['sku_age_class']] = config.AGE_CLASS[i][0]

            '''计算订单出库库龄'''
            order_df = df.groupby(['orderID'])['sku_age'].max().reset_index()
            order_df['order_age_class'] = ''
            for i in range(rank_num):
                order_df.loc[(order_df['sku_age'] > config.AGE_CLASS[i][1]) & (order_df['sku_age'] <= config.AGE_CLASS[i][2]), ['order_age_class']] = config.AGE_CLASS[i][0]

            order_df = order_df[['orderID', 'order_age_class']]
            df = pd.merge(df, order_df, on='orderID', how='left')

            self.sku_age_df = self.general_pivot(df, index=['sku_age_class'], pt_col=['quantity', 'sku'], distinct_count=['sku'], isCumu=True)
            self.order_age_df = self.general_pivot(df, index=['order_age_class', 're_order_structure'], pt_col=['orderID', 'quantity'], distinct_count=['orderID'], isCumu=True)

            self.ob_timeTableView.setModel(dfModel(self.sku_age_df))
            self.show_info_dialog('【SKU库龄】计算完成！')
        except:
            self.show_error_dialog('【SKU库龄】计算错误！！！请确认字段名是否输入正确！！！')

    def outbound_order_age_analysis(self):
        try:
            print('In outbound_order_age_analysis function!!!')
            self.ob_timeTableView.setModel(dfModel(self.order_age_df))
            self.show_info_dialog('【出库库龄】计算完成！')
        except:
            self.show_error_dialog('【出库库龄】计算错误！！！请确认字段名是否输入正确！！！')

    def outbound_sku_hour_in_analysis(self):
        try:
            print('In outbound_sku_hour_in_analysis function!!!')

            hour_in_col = ['time_in', 'orderID', 'sku', 'hour', 'quantity']
            df = self.ob_validDF[hour_in_col]
            self.sku_hour_in_df = self.hour_accumulate_sku(df=df)
            self.ob_timeTableView.setModel(dfModel(self.sku_hour_in_df))
            self.show_info_dialog('【SKU小时流入】计算完成！')
        except:
            self.show_error_dialog('【SKU小时流入】计算错误！！！请确认字段名是否输入正确！！！')

    def outbound_ABC_analysis(self):
        try:
            print('In outbound_ABC_analysis function!!!')
            ABC_interval_list = self.ob_ABCparameter.text().split(',')
            ob_ABCfreq_list = self.ob_ABCfreq.text().split(',')
            ob_ABCintervalDays_list = self.ob_ABCintervalDays.text().split(',')

            'ABC分析参数'
            ABC_interval = [float(x) for x in ABC_interval_list]
            ob_ABCfreq = [float(x) for x in ob_ABCfreq_list]
            ob_ABCintervalDays = [int(x) for x in ob_ABCintervalDays_list]

            print('ABC_interval: ', type(ABC_interval), ABC_interval)
            print('ob_ABC freq: ', type(ob_ABCfreq), ob_ABCfreq)
            print('ob_ABC intervalDays: ', type(ob_ABCintervalDays), ob_ABCintervalDays)

            # if ratio is None:
            #     ratio = [0.7, 0.2, 0.1]  # A类累计占比70%， B类累计占比70%~90%， C类累计占比90%~100%
            # if freq is None:
            #     freq = [0.5, 0.2]  # A类出库频次大于50%， B类出库频次为20%~50%， C类出库批次小于20%
            # if interval is None:
            #     interval = [3, 10]

            'ABC分析时间范围'
            ob_startDate = self.ob_startDate.text()
            ob_endDate = self.ob_endDate.text()

            # print('ob_startDate: ', type(ob_startDate), ob_startDate)
            # print('出库日期列类型： ', self.ob_validDF[['date']].dtypes)

            all_ob_dates = self.ob_validDF['date'].astype(np.str_).copy().unique()
            # print('all_ob_dates type ', type(all_ob_dates))
            # print('所有出库日期： ', all_ob_dates)

            ### ABC计算周期
            # 判断日期格式：
            if '/' in str(all_ob_dates) and '-' in ob_startDate:
                ob_startDate = ob_startDate.replace('-', '/')
                ob_endDate = ob_endDate.replace('-', '/')
            elif '-' in str(all_ob_dates) and '/' in ob_startDate:
                ob_startDate = ob_startDate.replace('/', '-')
                ob_endDate = ob_endDate.replace('/', '-')
            else:
                pass

            # print('ob_startDate &  ob_endDate', ob_startDate, ob_endDate)

            if ob_startDate in all_ob_dates and ob_endDate in all_ob_dates:
                print('起始和终止日期都有效')
                df = self.ob_validDF.query('date >= "{}" & date <= "{}"'.format(ob_startDate, ob_endDate))
            elif ob_startDate in all_ob_dates:
                print('起始日期都有效')
                df = self.ob_validDF.query('date >= "{}"'.format(ob_startDate))
            elif ob_endDate in all_ob_dates:
                print('终止日期都有效')
                df = self.ob_validDF.query('date <= "{}"'.format(ob_endDate))
            else:
                print('不筛选，以所有数据分析')
                df = self.ob_validDF

            self.abc_info = '\n 出库总天数: {}, \t 开始日期: {}, \t 结束日期： {}'.format(df['date'].nunique(), df['date'].min(), df['date'].max())
            print('筛选天数：', df['date'].nunique())
            print('起始日期：', df['date'].min())
            print('终止日期：', df['date'].max())

            ### 组合ABC参数
            multi_para = {'A': [1, 0],
                          'C': [0, 2]}  # 组合ABC=A表示：A的个数≥1，C的个数=0， 组合ABC=C表示：A的个数=0，C的个数≥2
            ### 出库件数ABC 和 出库频次ABC

            sku_df = pd.pivot_table(df,
                                    index=['sku'],
                                    values=['date', 'quantity'],
                                    aggfunc={'date': pd.Series.nunique, 'quantity': np.sum},
                                    fill_value=0).reset_index()

            ### 根据同一sku不同出库时间的库龄计算sku的加权库龄
            sku_age = pd.pivot_table(df, index=['sku'],
                                     values=['sku_age'],
                                     aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'quantity']),
                                     margins=False,
                                     fill_value=0).reset_index()

            sku_df = pd.merge(sku_df, sku_age, on=['sku'], how='left')

            sku_df = sku_df.sort_values(by='quantity', ascending=False)
            sku_df.columns = ['sku', 'ob_days', 'ob_quantity', 'sku_age']

            ''' 计算sku出库库龄等级'''
            config = Config()
            config.run()

            sku_df['sku_age_class'] = ''
            rank_num = len(config.AGE_CLASS)
            for i in range(rank_num):
                sku_df.loc[(sku_df['sku_age'] > config.AGE_CLASS[i][1]) & (sku_df['sku_age'] <= config.AGE_CLASS[i][2]), ['sku_age_class']] = config.AGE_CLASS[i][0]

            '''计算出库件数ABC'''
            sku_df['cumu_qty'] = sku_df['ob_quantity'].cumsum()
            sku_df['cumu_qty%'] = sku_df['cumu_qty'] / sku_df['ob_quantity'].sum()

            # 计算ABC
            sku_df['qty_ABC'] = 'C'
            sku_df.loc[(sku_df['cumu_qty%'] <= ABC_interval[1]), ['qty_ABC']] = 'B'
            sku_df.loc[(sku_df['cumu_qty%'] <= ABC_interval[0]), ['qty_ABC']] = 'A'

            print('件数ABC计数： ', sku_df['qty_ABC'].value_counts())

            index_ABC = ''
            ### 判断是否计算组合ABC
            # if multiple == False:
            #     single_col = ['sku', 'ob_quantity', 'cumu_qty%', 'qty_ABC']
            #     sku_df = sku_df[single_col]
            #     index_ABC = 'qty_ABC'

            ''' 计算出库频次ABC '''
            print('00000 计算出库频次ABC')
            print('sku_df: ', sku_df)
            total_day = df['date'].nunique()
            sku_df['freq_day'] = sku_df['ob_days'] / total_day

            # 计算ABC
            sku_df['freq_ABC'] = 'B'
            sku_df.loc[(sku_df['freq_day'] >= ob_ABCfreq[0]), ['freq_ABC']] = 'A'
            sku_df.loc[(sku_df['freq_day'] < ob_ABCfreq[1]), ['freq_ABC']] = 'C'

            '''计算动销间隔天数ABC'''
            print('11111 计算动销间隔天数ABC')
            temp_df = df[['sku', 'date']]
            temp_df = temp_df.sort_values(by=['sku', 'date'])
            sku_groups = temp_df.groupby(['sku'])

            sku_interval_day_list = []
            for sku, group in sku_groups:
                group['interval_days'] = pd.to_timedelta(group['date'].shift(-1) - group['date']).dt.days
                max_interval_days = group['interval_days'].max()  # 最大间隔天数
                sku_interval_day_list.append([sku, max_interval_days])

            sku_interval_day_df = pd.DataFrame(sku_interval_day_list, columns=['sku', 'max_interval_days'])

            ### 计算间隔天数ABC
            sku_interval_day_df['interval_ABC'] = 'B'

            sku_interval_day_df.loc[(sku_interval_day_df['max_interval_days'] < ob_ABCintervalDays[0]), ['interval_ABC']] = 'A'
            sku_interval_day_df.loc[(sku_interval_day_df['max_interval_days'] > ob_ABCintervalDays[1]), ['interval_ABC']] = 'C'

            sku_df = pd.merge(sku_df, sku_interval_day_df, on='sku', how='left')

            '''
            计算3个维度的组合ABC
            '''
            print('22222 计算3个维度的组合ABC')

            ABC_col = ['qty_ABC', 'freq_ABC', 'interval_ABC']
            abc = ['A', 'B', 'C']
            # 统计3个维度ABC的个数
            for item in abc:
                sku_df[item + '_cnt'] = sum([sku_df[col].map(lambda x: x.count(item)) for col in ABC_col])

            # 根据ABC的个数计算组合ABC

            sku_df['combine_ABC'] = 'B'
            sku_df.loc[(sku_df['A_cnt'] >= multi_para['A'][0]) & (sku_df['C_cnt'] == multi_para['A'][1]), ['combine_ABC']] = 'A'
            sku_df.loc[(sku_df['A_cnt'] == multi_para['C'][0]) & (sku_df['C_cnt'] >= multi_para['C'][1]), ['combine_ABC']] = 'C'

            print('22222 sku_df: ', sku_df)

            '''根据当前是否有库存数据，汇总ABC分类数据'''
            if self.inv_df.shape[0]>0:

                multi_col = ['sku', 'ob_quantity', 'cumu_qty', 'cumu_qty%', 'qty_ABC', 'ob_days', 'freq_day', 'freq_ABC', 'max_interval_days',
                             'interval_ABC', 'combine_ABC']
                sku_df = sku_df[multi_col]
                print('22222 sku_df: ', sku_df)

                index_ABC = 'combine_ABC'

                ### 库存sku库龄分析, 存在库存维度的sku分析
                inv_skuCol = self.inv_df.columns[int(self.inv_skuCol.text())]  # 产品代码
                inv_invQty = self.inv_df.columns[int(self.inv_invQty.text())]  # 在库件数
                inv_invVol = self.inv_df.columns[int(self.inv_invVol.text())]  # 在库体积
                inv_sku_df = self.inventory_get_sku_pivot(inv_skuCol, inv_invQty, inv_invVol)

                print('inventory sku pivot!!!!!!!!!!!!!!')
                print(inv_sku_df.columns)
                print(inv_sku_df.shape)
                print(inv_sku_df.head(10))

                sku_ABC_detail = pd.merge(sku_df, inv_sku_df, on=['sku'], how='left')
                print('outbound sku pivot!!!!!!!!!!!!!!')
                print(sku_df.columns)
                print(sku_df['combine_ABC'].value_counts())

                print('test！！！！！！！！！！！')
                print(sku_ABC_detail.columns)
                print(sku_ABC_detail.shape)

                index = [index_ABC] + ['age_class']
                sku_ABC_df = pd.pivot_table(sku_ABC_detail,
                                            index=index,
                                            values=['ob_quantity', 'inv_quantity', 'sku'],
                                            aggfunc={'ob_quantity': np.sum, 'inv_quantity': np.sum, 'sku': pd.Series.nunique},
                                            margins=True,
                                            fill_value=0).reset_index()
                print('*******', sku_ABC_df.dtypes)
                # print(sku_ABC_df)

                self.sku_ABC_df = sku_ABC_df
                self.sku_ABC_detail = sku_ABC_detail

                self.ob_timeTableView.setModel(dfModel(self.sku_ABC_df))
                self.show_info_dialog('【ABC分析】计算完成！')
            else:
                index_ABC = 'qty_ABC'

                index = [index_ABC] + ['sku_age_class']
                sku_ABC_df = pd.pivot_table(sku_df,
                                            index=index,
                                            values=['ob_quantity', 'sku'],
                                            aggfunc={'ob_quantity': np.sum, 'sku': pd.Series.nunique},
                                            margins=True,
                                            fill_value=0).reset_index()

                self.sku_ABC_df = sku_ABC_df
                self.sku_ABC_detail = sku_df

                print('&&&&&&&&&', sku_ABC_df.dtypes)
                self.ob_timeTableView.setModel(dfModel(self.sku_ABC_df))
                self.show_info_dialog('【ABC分析】计算完成！')
        except:
            self.show_error_dialog('【ABC分析】计算错误！！！请确认分类参数及字段名是否输入正确！！！')

    def outbound_channel_analysis(self):
        try:
            print('In outbound_channel_analysis function!!!')
            sample_cols = ['date', 'orderID', 'sku', 'sku_size', 'quantity', 'customer', 'channel', 'platform']
            channel_index = ['channel']
            sample_df = self.ob_validDF[sample_cols]

            self.channel_order_distribution = self.order_distribution(df = sample_df, index=channel_index)
            # 按 qty 降序排列
            self.channel_order_distribution = self.channel_order_distribution.sort_values(by='qty', ascending=False, ignore_index=True)

            self.ob_timeTableView.setModel(dfModel(self.channel_order_distribution))
            self.show_info_dialog('【渠道分析】计算完成！')
        except:
            self.show_error_dialog('【渠道分析】计算错误！！！请确认字段名是否输入正确！！！')



    def outbound_platform_analysis(self):
        try:
            print('In outbound_platform_analysis function!!!')
            sample_cols = ['date', 'orderID', 'sku', 'sku_size', 'quantity', 'customer', 'channel', 'platform']
            platform_index = ['platform']
            sample_df = self.ob_validDF[sample_cols]
            self.platform_order_distribution = self.order_distribution(df=sample_df, index=platform_index)
            # 按 qty 降序排列
            self.channel_order_distribution = self.channel_order_distribution.sort_values(by='qty', ascending=False, ignore_index=True)

            self.ob_timeTableView.setModel(dfModel(self.platform_order_distribution))
            self.show_info_dialog('【平台分析】计算完成！')
        except:
            self.show_error_dialog('【平台分析】计算错误！！！请确认字段名是否输入正确！！！')

    def outbound_download_all_results(self):
        print('in outbound_download_all_results function!')
        try:
            filePath, ok2 = QFileDialog.getSaveFileName(None, caption='保存文件', filter='Excel files(*.xlsx , *.xls)')

            if 'xls' in filePath or 'xlsx' in filePath:
                ### write to file
                writer = pd.ExcelWriter(filePath)

                outbound_source_info = '数据源\n 原始数据: 行数 {}, \t列数 {}'.format(self.ob_validDF.shape[0], self.ob_validDF.shape[1])
                print('outbound data info: ', outbound_source_info)

                #EIQ分析
                if self.normal_order_EIQ.shape[0]>0:
                    self.format_data(writer=writer, df=self.normal_order_EIQ, sheet_name='B1.1-标准订单EIQ', source_data_info=outbound_source_info)
                if self.fba_order_EIQ.shape[0] > 0:
                    self.format_data(writer=writer, df=self.fba_order_EIQ, sheet_name='B1.2-FBA EIQ', source_data_info=outbound_source_info)
                if self.order_type_EIQ.shape[0] > 0:
                    self.format_data(writer=writer, df=self.order_type_EIQ, sheet_name='B1.3-订单结构EIQ', source_data_info=outbound_source_info)
                if self.wave_order_type_EIQ.shape[0] > 0:
                    self.format_data(writer=writer, df=self.wave_order_type_EIQ, sheet_name='B1.4-波次EIQ', source_data_info=outbound_source_info)
                if self.current_pick_order_EIQ.shape[0] > 0:
                    self.format_data(writer=writer, df=self.current_pick_order_EIQ, sheet_name='B1.5-拣货EIQ', source_data_info=outbound_source_info)

                ###多品订单组合分析
                if self.multi_order_EIQ.shape[0] > 0:
                    self.format_data(writer=writer, df=self.multi_order_EIQ, sheet_name='B2-多品订单组合', source_data_info=outbound_source_info)

                ### 客户动销分析
                # print('self.customer_sales.shape : ', self.customer_sales.shape)
                if self.customer_sales.shape[0] > 0:
                    self.format_data(writer=writer, df=self.customer_sales, sheet_name='B3-客户动销', source_data_info=outbound_source_info)


                ### SKU动销
                # print('self.sku_active_sales.shape : ', self.sku_active_sales.shape)
                if self.sku_active_sales.shape[0] > 0:
                    self.format_data(writer=writer, df=self.sku_active_sales, sheet_name='B4-SKU动销', source_data_info=outbound_source_info)

                ### SKU小时流入
                if self.sku_hour_in_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.sku_hour_in_df, sheet_name='B5-SKU小时流入', source_data_info=outbound_source_info)


                ### ABC分析
                # print('self.sku_ABC_df.shape : ', self.sku_ABC_df.shape)
                if self.sku_ABC_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.sku_ABC_df, sheet_name='B6.1-ABC分类', source_data_info=outbound_source_info + self.abc_info)

                if self.sku_ABC_detail.shape[0] > 0:
                    self.format_data(writer=writer, df=self.sku_ABC_detail, sheet_name='B6.2-SKU ABC分类明细', source_data_info=outbound_source_info + self.abc_info)

                ### SKU库龄分析
                # print('self.sku_age_df.shape : ', self.sku_age_df.shape)
                if self.sku_age_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.sku_age_df, sheet_name='B7.1-SKU出库库龄', source_data_info=outbound_source_info)

                ### 订单库龄分析
                # print('self.order_age_df.shape : ', self.order_age_df.shape)
                if self.order_age_df.shape[0] > 0:
                    self.format_data(writer=writer, df=self.order_age_df, sheet_name='B7.2-订单出库库龄', source_data_info=outbound_source_info)

                ### 渠道分析
                # print('self.channel_order_distribution.shape : ', self.channel_order_distribution.shape)
                if self.channel_order_distribution.shape[0] > 0:
                    self.format_data(writer=writer, df=self.channel_order_distribution, sheet_name='B8.1-渠道分布', source_data_info=outbound_source_info)

                ### 平台分析
                #print('self.platform_order_distribution.shape : ', self.platform_order_distribution.shape)
                if self.platform_order_distribution.shape[0] > 0:
                    self.format_data(writer=writer, df=self.platform_order_distribution, sheet_name='B8.2-平台分布', source_data_info=outbound_source_info)

                writer.save()
                self.show_info_dialog('出库分析结果保存成功！')

            else:
                self.show_info_dialog('请保存为指定的文件类型！')
        except:
            self.show_error_dialog('文件保存失败！！！')



    def outbound_clear_all(self):
        try:
            print('In outbound_clear_all function!!!')
            self.ob_tableView.setModel(dfModel(pd.DataFrame()))
            self.ob_timeTableView.setModel(dfModel(pd.DataFrame()))

            self.ob_totalRow.clear()  # 数据总行数
            self.ob_errorRow.clear()  # 异常数据
            self.ob_validRow.clear()  # 有效数据总行数
            self.ob_totalOrders.clear()  # 总订单数
            self.ob_totalSKU.clear()  # 总SKU数
            self.ob_totalDays.clear()  # 总出库天数

        except:
            self.show_error_dialog('清理数据失败！')

    def get_EIQ(self, df, index=None, isPercentage=False):
        ### EIQ的最小单位为日期
        try:
            if index is None:
                pt_index = ['month', 'week', 'weekday', 'date']
            else:
                pt_index = ['month', 'week', 'weekday', 'date'] + index

            date_EIQ = pd.pivot_table(df, index=pt_index,
                                      values=['orderID', 'quantity', 'sku', 'sku_size', ],
                                      aggfunc={'orderID': pd.Series.nunique, 'quantity': sum, 'sku': pd.Series.nunique, 'sku_size': len},
                                      fill_value=0).reset_index()

            print('date_EIQ columns: ', date_EIQ.columns)

            date_EIQ.columns = pt_index + ['order', 'qty', 'sku', 'line']

            # 按index的维度，增加均值和峰值
            order_columns = ['order', 'sku', 'line', 'qty']
            # eiq_col = ['EN', 'EQ', 'IK', 'IQ']

            date_EIQ = date_EIQ[pt_index + order_columns]
            date_EIQ = date_EIQ.sort_values(by=['date'] + index)

            ### 汇总的维度
            marge_index = pt_index[-1]

            avg_value = date_EIQ.groupby(marge_index)[order_columns].mean().reset_index()
            avg_value['date'] = 'Average'

            max_value = date_EIQ.groupby(marge_index)[order_columns].max().reset_index()
            max_value['date'] = 'Max'

            # 日均值行
            date_EIQ = pd.concat([date_EIQ, avg_value], ignore_index=True)

            # 日峰值行
            date_EIQ = pd.concat([date_EIQ, max_value], ignore_index=True)

            print(date_EIQ.columns)
            print(date_EIQ.head(10))

            ### 计算日维度的EIQ
            date_EIQ['EN'] = date_EIQ['line'] / date_EIQ['order']
            date_EIQ['EQ'] = date_EIQ['qty'] / date_EIQ['order']
            date_EIQ['IK'] = date_EIQ['line'] / date_EIQ['sku']
            date_EIQ['IQ'] = date_EIQ['line'] / date_EIQ['sku']

            date_EIQ['qty/line'] = date_EIQ['qty'] / date_EIQ['line']
            # print('re_EIQ.columns: ', date_EIQ.columns)

            if isPercentage:
                sum_col = ['order', 'sku', 'line', 'qty']
                date_df = date_EIQ.groupby('date')[sum_col].sum().reset_index()  # sku累加

                date_df = date_EIQ.groupby('date')[sum_col].sum().reset_index()

                re_sum_col = ['sum_order', 'sum_sku', 'sum_line', 'sum_qty']
                # 修改日维度的列
                date_df.columns = ['date'] + re_sum_col

                date_EIQ = pd.merge(date_EIQ, date_df, on='date', how='left')
                for col in sum_col:
                    date_EIQ[col + '%'] = date_EIQ[col] / date_EIQ['sum_' + col]

                date_EIQ = date_EIQ.drop(re_sum_col, axis=1)  # 删除日维度列

            return date_EIQ
        except:
            self.show_error_dialog('【出库分析】EIQ计算失败！')

    def get_pick_EIQ(self, df, index):
        ### EIQ的最小单位为日期
        try:
            if index is None:
                pt_index = ['month', 'week', 'weekday', 'date']
            else:
                pt_index = ['month', 'week', 'weekday', 'date'] + index

            date_EIQ = pd.pivot_table(df, index=pt_index,
                                      values=['orderID', 'quantity', 'sku', 'sku_size', ],
                                      aggfunc={'orderID': pd.Series.nunique, 'quantity': sum, 'sku': pd.Series.nunique, 'sku_size': len},
                                      fill_value=0).reset_index()
            date_EIQ.columns = pt_index + ['order', 'qty', 'sku', 'line']  # 重命名列
            order_columns = ['order', 'sku', 'line', 'qty']
            date_EIQ = date_EIQ[pt_index + order_columns]  # 列重排列

            pick_columns = ['date', 'pickupNO', 'location', 'sku']
            pick_temp = pd.pivot_table(df[pick_columns], index=['date', 'pickupNO'],
                                       values=['location', 'sku'],
                                       aggfunc={'location': pd.Series.nunique, 'sku': pd.Series.nunique},
                                       fill_value=0).reset_index()
            pick_pt = pd.pivot_table(pick_temp, index=['date'],
                                     values=['location', 'pickupNO', 'sku'],
                                     aggfunc={'location': np.sum, 'pickupNO': len, 'sku': np.sum},
                                     fill_value=0).reset_index()
            pick_pt.columns = ['date', 'location', 'pickupNO', 'sku_touch_cnt']
            pick_pt = pick_pt[['date', 'pickupNO', 'location', 'sku_touch_cnt']]

            re_pt = pd.merge(date_EIQ, pick_pt, on='date', how='left')

            # 订单维度EIQ
            re_pt['EN'] = re_pt['line'] / re_pt['order']
            re_pt['EQ'] = re_pt['qty'] / re_pt['order']
            re_pt['IK-order'] = re_pt['line'] / re_pt['sku']  # 订单维度IK
            re_pt['IK-pick'] = re_pt['location'] / re_pt['sku']  # 拣货维度IK
            re_pt['IQ'] = re_pt['qty'] / re_pt['sku']

            re_pt['qty/line-order'] = re_pt['qty'] / re_pt['line']  # 订单维度行均件
            re_pt['qty/line-pick'] = re_pt['qty'] / re_pt['location']  # 拣货维度行均件

            re_pt = re_pt.sort_values(by=pt_index)  # 按透视字段排序

            return re_pt
        except:
            self.show_error_dialog('【出库分析】EIQ计算失败！')

    def order_distribution(self, df, index):
        try:
            re = pd.pivot_table(df, index=index,
                                values=['orderID', 'quantity', 'sku', 'sku_size'],  # 行数计数任选一列，此处选择的是sku_size
                                aggfunc={'orderID': pd.Series.nunique, 'quantity': sum, 'sku': pd.Series.nunique, 'sku_size': len},
                                margins=True,
                                margins_name='总计',
                                fill_value=0).reset_index()

            re.columns = index + ['order', 'qty', 'sku', 'line']
            order_columns = ['order', 'sku', 'line', 'qty']
            re = re[index + order_columns]
            # 计算比例
            for i in range(len(order_columns)):
                re[order_columns[i] + '%'] = re[order_columns[i]] / (re[order_columns[i]].sum() / 2)
            re = re.sort_values(by=index, ascending=True, ignore_index=True)
            return re
        except:
            self.show_error_dialog('【出库分析】订单维度计算失败！')

    def general_pivot(self, df, index, pt_col, distinct_count=None, isCumu=False):
        """
        :param df: 透视表原始数据
        :param index: 透视的行
        :param distinct_count: 是否添加 SKU_ID 字段
        :param pt_col: 透视的 values，即透视字段
        :param isCumu: 默认为False, 是否计算累计比例
        :return:
        """
        try:
            col_function = {}
            for col in pt_col:
                f = np.sum
                if distinct_count is not None and col in distinct_count:
                    f = pd.Series.nunique
                col_function[col] = f
            result_pt = pd.pivot_table(df, index=index,
                                       values=pt_col,
                                       aggfunc=col_function,
                                       fill_value=0).reset_index()
            index_num = len(index)
            cols = list(result_pt.columns[index_num:])

            # 透视字段， 需计算总和及百分比的字段
            sum_col = pt_col

            row_n = result_pt.shape[0]  #总计行
            # 更新合计值
            result_pt.at[row_n, index[-1:]] = '合计'
            result_pt.at[row_n, sum_col] = result_pt[sum_col].apply(lambda x: x.sum())

            # 计算比例
            for i in range(len(sum_col)):
                result_pt[sum_col[i] + '%'] = result_pt[sum_col[i]] / (result_pt[sum_col[i]].sum() / 2)

            # 判断是否计算累计比例，若计算，一般为件数及体积的累计比例
            if isCumu:
                for i in range(len(cols)):
                    result_pt['cumu_' + cols[i] + '%'] = result_pt[cols[i] + '%'].cumsum()
                    result_pt.at[row_n, 'cumu_' + cols[i] + '%'] = np.NAN
            return result_pt
        except:
            self.show_error_dialog('【出库分析】基础计算失败！')

    def hour_accumulate_sku(self, df):
        '''
        计算给定日期的订单累计流入
        :param df: 必须包含time_in（或date，hour）列，订单号，sku,件数字段 ['time_in', 'orderID', 'sku', 'hour', 'quantity']
        :param date: 日期筛选参数，兼容string和list类型
        :return:
        '''
        try:
            # 从流入时间字段中截取日期
            date = self.ob_hourinDate.text()
            cutoff_hour = int(self.ob_hourinCutoffHour.text())

            print('DATE: ', date, 'DATE TYPE: ', type(date))
            print('cutoff_hour: ', cutoff_hour, 'cutoff_hour TYPE: ', type(cutoff_hour))

            df['date'] = pd.to_datetime(df['time_in'].dt.date)

            # print(df.head(10))

            df['process_date'] = df['date'].copy()
            print(df.dtypes)
            print(df.head(10))

            df['day_delta'] = pd.Timedelta(days=0)
            df.loc[(df['hour'] >= cutoff_hour), ['day_delta']] = pd.Timedelta(days=1)

            ### 将日期列转化为float，进行加减
            df['process_date'] = df['process_date'].values.astype(float)
            df['day_delta'] = df['day_delta'].values.astype(float)

            # 更新截单点后的操作日期
            df.loc[(df['hour'] >= cutoff_hour), ['process_date']] = pd.to_datetime(df['process_date'] + df['day_delta'])
            df['process_date'] = pd.to_datetime(df['process_date'])
            df = df.drop(columns='day_delta')  # 删除计算操作日期的辅助列

            ### 计算累计订单是的index
            df['index_hour'] = df['hour'].apply(lambda x: x - cutoff_hour if x >= cutoff_hour else x + (24 - cutoff_hour))

            if type(date) is np.str:
                df = df.query('process_date == "{}"'.format(date))
            elif type(date) is list:
                df = df.loc[df['process_date'].isin(date)]
            else:
                print("请输入正确的日期参数！")
                print("形如 "'"2022-01-01"'" 或 ["'"2022-01-01"'", "'"2022-01-10"'", "'"2022-01-15"'"]")

            print('订单累计流入数据源')
            print(df.dtypes)
            print(df.head(10))

            '''
            初始化结果dataframe
            '''
            re_col = ['date', 'curr_hour', 'next_hour', 'local_hour', 'order_cnt', 'line_cnt', 'sku_cnt', 'qty', 'in_sku', 'out_sku',
                      'cumu_order_cnt', 'cumu_line_cnt', 'cumu_sku_cnt', 'cumu_qty',
                      'cumu_order_cnt%', 'cumu_line_cnt%', 'cumu_sku_cnt%', 'cumu_qty%']
            re_df = pd.DataFrame(columns=re_col)

            ## 按操作日期分组
            df['process_date'] = df['process_date'].astype(np.str).copy()  # process_date作为groupby的index，转化为string更简洁

            date_groups = df.groupby(['process_date'])
            print('按操作日期分组 date_groups: ', date_groups)

            for day, date_df in date_groups:
                hour_groups = date_df.groupby(['index_hour'])

                hour_dict = {}
                hour_list = []
                for h, hour_df in hour_groups:
                    hour_dict[h] = hour_df

                for i in range(24):
                    curr_hour = i
                    next_hour = i + 1

                    if curr_hour >= cutoff_hour:
                        local_hour = curr_hour - cutoff_hour
                    else:
                        local_hour = curr_hour + cutoff_hour

                    ### current hour order discrible

                    # 若当前小时没有订单数据，给curr_df赋值0，订单描述性参数都为0
                    curr_df = hour_dict.get(curr_hour, None)
                    next_df = hour_dict.get(next_hour, None)
                    if curr_df is not None:
                        order_cnt = curr_df['orderID'].nunique()
                        line_cnt = curr_df.shape[0]
                        sku_cnt = curr_df['sku'].nunique()
                        qty = curr_df['quantity'].sum()
                    else:
                        order_cnt = 0
                        line_cnt = 0
                        sku_cnt = 0
                        qty = 0

                    ### 统计当前小时的sku流入流出
                    if curr_df is not None and next_df is not None:
                        in_cnt = len(set(next_df['sku'].unique()).difference(set(curr_df['sku'].unique())))
                        out_cnt = len(set(curr_df['sku'].unique()).difference(set(next_df['sku'].unique())))
                    elif curr_df is not None:
                        in_cnt = 0
                        out_cnt = sku_cnt
                    elif next_df is not None:
                        in_cnt = next_df['sku'].nunique()
                        out_cnt = 0
                    else:
                        in_cnt = 0
                        out_cnt = 0

                    ### accumulate order discrible
                    # 若累计到当前小时没有订单数据，给cumu_df赋值0，订单描述性参数都为0
                    cumu_df = date_df[date_df['index_hour'] <= curr_hour]
                    if cumu_df.shape[0] != 0:
                        cumu_order_cnt = cumu_df['orderID'].nunique()
                        cumu_line_cnt = cumu_df.shape[0]
                        cumu_sku_cnt = cumu_df['sku'].nunique()
                        cumu_qty = cumu_df['quantity'].sum()
                    else:
                        cumu_order_cnt = 0
                        cumu_line_cnt = 0
                        cumu_sku_cnt = 0
                        cumu_qty = 0

                    hour_list.append([day, curr_hour, next_hour, local_hour, order_cnt, line_cnt, sku_cnt, qty, in_cnt, out_cnt,
                                      cumu_order_cnt, cumu_line_cnt, cumu_sku_cnt, cumu_qty])

                hour_accu_col = ['date', 'curr_hour', 'next_hour', 'local_hour', 'order_cnt', 'line_cnt', 'sku_cnt', 'qty', 'in_sku', 'out_sku',
                                 'cumu_order_cnt', 'cumu_line_cnt', 'cumu_sku_cnt', 'cumu_qty']
                hour_accu_df = pd.DataFrame(hour_list, columns=hour_accu_col)

                print('000000000 hour_accu_df: ', hour_accu_df)

                ### 增加累计比例列
                accu_percentage_col = ['order_cnt', 'line_cnt', 'sku_cnt', 'qty']
                for i in accu_percentage_col:
                    if i == 'sku_cnt':
                        total_sku = hour_accu_df.tail(1)['cumu_' + i].to_list()
                        hour_accu_df['cumu_' + i + '%'] = hour_accu_df['cumu_' + i] / total_sku
                    else:
                        hour_accu_df['cumu_' + i + '%'] = hour_accu_df['cumu_' + i] / hour_accu_df[i].sum()

                re_df = re_df.append(hour_accu_df)
                print('1111111111 re_df: ', re_df)

            # print('='*10, '订单累计流入结果')
            # print(re_df.dtypes)
            # print(re_df)

            return re_df
        except:
            self.show_error_dialog('【SKU小时流入】分析计算错误！！！请确认出库数据及字段名是否输入正确,时间及截单时间参数是否输入正确！！！')



    '''
    入库分析 事件函数
    '''
    def inbound_load_data(self):
        try:
            filenames = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls, *.csv)')
            filename = filenames[0]

            if 'csv' in filename:
                try:
                    self.ib_df = pd.read_csv(filename, encoding='utf-8')
                except:
                    self.ib_df = pd.read_csv(filename, encoding='gbk')
                # 删除有空值的行
                row1 = self.ib_df.shape[0]
                self.ib_df.dropna(how='all', inplace=True)
                row2 = self.ib_df.shape[0]
                self.ib_totalRow.setText("{:,}".format(row2))  # 数据总行数
                self.ib_errorRow.setText("{:,}".format(row1 - row2))  # 异常数据
                model = dfModel(self.ib_df.head(100))
                self.ib_dataTableView.setModel(model)
            elif 'xlsx' in filename:
                self.ib_df = pd.read_excel(filename)
                # 删除有空值的行
                row1 = self.ib_df.shape[0]
                self.ib_df.dropna(how='all', inplace=True)
                row2 = self.ib_df.shape[0]
                self.ib_totalRow.setText("{:,}".format(row2))  # 数据总行数
                self.ib_errorRow.setText("{:,}".format(row1 - row2))  # 异常数据
                model = dfModel(self.ib_df.head(100))
                self.ib_dataTableView.setModel(model)
            else:
                self.show_error_dialog('请选择csv或xlsx文件类型!')
        except:
            self.show_error_dialog('请选择文件!')

    def inbound_data_process(self):
        try:
            print('In inbound_data_process function!!!')
            # ### 交互界面，输入EIQ分析的有效字段编号
            # print('\n')
            # print('请按以下字段顺序输入 入库明细 对应的列号：（列号从0开始，以空格隔开以enter结束）')
            # print('收货日期 最早签收日期 入库单号 入库单类型 货运方式 海柜号 跟踪号 客户代码 箱号 产品代码 产品货型 收货数量 收货体积 物理仓编码')
            #
            # # column_index = [int(x) for x in input().split()]
            # detail_index = [16, 8, 6, 7, 3, 14, 20, 10, 17, 4, 19, 24, 21, 15]
            # column_name = data.columns.tolist()

            print('出库数据1： ', self.ib_df.shape)
            row1 = self.ib_df.shape[0]

            # print('column_index: ', detail_index)
            date = self.ib_df.columns[int(self.ib_dateCol.text())]                   # 收货日期
            receive_time = self.ib_df.columns[int(self.ib_receiveTimeCol.text())]    # 最早签收日期
            inboundID = self.ib_df.columns[int(self.ib_inboundIdCol.text())]         # 入库单号
            inbound_type = self.ib_df.columns[int(self.ib_inboundTypeCol.text())]    # 入库单类型
            delivery_mode = self.ib_df.columns[int(self.ib_deliveryModeCol.text())]  # 货运方式
            containerNO = self.ib_df.columns[int(self.ib_containerNoCol.text())]     # 海柜号
            trackingNO = self.ib_df.columns[int(self.ib_trackingNO.text())]          # 跟踪号
            cartonNO = self.ib_df.columns[int(self.ib_cartonNoCol.text())]           # 箱号
            wh_code = self.ib_df.columns[int(self.ib_warehouseCodeCol.text())]       # 物理仓
            customer = self.ib_df.columns[int(self.ib_customerCol.text())]           # 客户代码
            sku = self.ib_df.columns[int(self.ib_skuCol.text())]                     # 产品代码
            sku_size = self.ib_df.columns[int(self.ib_skuSizeCol.text())]            # 产品货型
            quantity = self.ib_df.columns[int(self.ib_quantityCol.text())]           # 收货数量
            vol = self.ib_df.columns[int(self.ib_volCol.text())]                     # 收货体积


            valid_data =self.ib_df[[date, receive_time, inboundID, inbound_type,delivery_mode,containerNO,trackingNO,cartonNO,
                                   wh_code,customer,sku,sku_size,quantity,vol]]
            print('inbound valid_data columns: ', valid_data.columns)

            valid_columns_name = ['date', 'receive_time', 'inboundID', 'inbound_type', 'delivery_mode', 'containerNO', 'trackingNO', 'cartonNO',
                                  'wh_code', 'customer',  'sku', 'sku_size', 'quantity', 'vol']
            valid_data.columns = valid_columns_name

            ### 日期列转化为Python日期格式

            valid_data.loc[:,'receive_time'] = pd.to_datetime(valid_data['receive_time'])
            valid_data.loc[:, 'date'] =valid_data['receive_time'].dt.date
            valid_data.loc[:, 'receive_date'] = valid_data['receive_time'].dt.date
            valid_data.loc[:, 'month'] = valid_data['receive_time'].dt.month
            valid_data.loc[:, 'weekday'] = valid_data['receive_time'].dt.weekday + 1

            valid_data.loc[:, 'deliveryNO'] = valid_data['containerNO']
            valid_data.loc[(valid_data['containerNO'].str.len() <= 1), ['deliveryNO']] = valid_data['trackingNO']

            print(valid_data.dtypes)

            self.ib_df = valid_data

            # 删除有空值的行
            self.ib_df.dropna(how='any', inplace=True)

            print('出库数据2： ', self.ib_df.shape)
            row2 = self.ib_df.shape[0]

            self.ib_totalRow.setText("{:,}".format(row1))  # 原始数据总行数
            self.ib_totalCol.setText("{:,}".format(self.ib_df.shape[1]))  # 原始数据总列数

            self.ib_errorRow.setText("{:,}".format(row1 - row2))  # 异常数据
            self.ib_validRow.setText("{:,}".format(row2))  # 有效数据总行数

            self.ib_totalSKU.setText("{:,}".format(self.ib_df['sku'].nunique()))  # 总sku数
            self.ib_totalDays.setText("{:,}".format(self.ib_df['receive_date'].nunique()))  # 总天数

            self.ib_dataTableView.setModel(dfModel(self.ib_df.head(100)))
            self.show_info_dialog('入库【数据处理】完成！')

        except:
            self.show_error_dialog('入库数据处理失败,请输入正确的字段列号!!!')


    def inbound_analysis(self):
        try:
            print('In inbound_analysis function!!!')
            index = ['month', 'date', 'delivery_mode']

            ib_start_date = self.ib_startDate.text()
            ib_end_date = self.ib_endDate.text()

            df = self.ib_df.query(' receive_time >= "{}" & receive_time <= "{}"'.format(ib_start_date, ib_end_date))

            sort_size = ['XL', 'L2', 'L1', 'M', 'S', 'XS']
            df['sku_size'] = pd.Categorical(df['sku_size'], sort_size)

            ## 客户在库体积及在库件数
            date_distribution_df = pd.pivot_table(df, index=index,
                                         values=['vol', 'quantity'],
                                         columns=['sku_size'],
                                         aggfunc='sum',
                                         margins=True,
                                         margins_name='All',
                                         fill_value=0).reset_index()

            ## 按客户总体积排序
            date_distribution_df = date_distribution_df.sort_values(by=('vol', 'All'), ascending=False, ignore_index=True)

            ### 多级索引转成单层索引
            col = []
            for (s1, s2) in date_distribution_df.columns:
                if len(s2) > 0:
                    col.append(s1 + '_' + str(s2))
                else:
                    col.append(s1)
            # delivery_df.columns = [ s1 + '_' + str(s2) for (s1, s2) in delivery_df.columns]
            date_distribution_df.columns = col

            ## 计算体积货型占比
            for item in sort_size:
                date_distribution_df[('vol_{}%'.format(item))] = date_distribution_df[('vol_{}'.format(item))] / date_distribution_df[('vol_All')]

            ## 计算件数货型占比
            for item in sort_size:
                date_distribution_df[('quantity_{}%'.format(item))] = date_distribution_df[('quantity_{}'.format(item))] / date_distribution_df[('quantity_All')]

            date_distribution_df = date_distribution_df.sort_values(by=['delivery_mode', 'date'], ignore_index=True)

            # print(date_distribution_df.dtypes)
            # print('delivery_df.shape ', date_distribution_df.shape)
            # print(date_distribution_df.head(5))

            self.ib_date_distribution = date_distribution_df

            self.ib_resultsTableView.setModel(dfModel(self.ib_date_distribution))
            self.show_info_dialog('【入库量分析】计算完成！')
        except:
            self.show_error_dialog('【入库量分析】计算错误,请确认字段列编号是否正确? 分析时间段与原始数据是否有重叠!')


    def inbound_carton_distribution(self):
        try:
            print('in inbound_carton_distribution function!!!')
            index = ['month', 'weekday', 'date', 'cartonNO']

            ib_start_date = self.ib_startDate.text()
            ib_end_date = self.ib_endDate.text()

            df = self.ib_df.query('receive_time >= "{}" & receive_time <= "{}"'.format(ib_start_date, ib_end_date))

            carton_df = pd.pivot_table(df, index=index,
                                       values=['quantity', 'sku'],
                                       aggfunc={'quantity': np.sum, 'sku': pd.Series.nunique},
                                       fill_value=0).reset_index()

            carton_df['carton_type'] = '异常'

            carton_df.loc[(carton_df['quantity'] == 1) & (carton_df['sku'] == 1), ['carton_type']] = '单箱单件'
            carton_df.loc[(carton_df['quantity'] > 1) & (carton_df['sku'] == 1), ['carton_type']] = '单箱单品'
            carton_df.loc[(carton_df['quantity'] > 1) & (carton_df['sku'] > 1), ['carton_type']] = '单箱多品'

            print('carton_type value count: ', df)

            carton_df['sku_type'] = carton_df['sku'].astype(np.str) + 'SKU'
            carton_df.loc[(carton_df['sku'] > 10), ['sku_type']] = '>10SKU'

            marge_index = ['cartonNO', 'carton_type', 'sku_type']

            df = pd.merge(df, carton_df[marge_index], on=['cartonNO'], how='left')

            print('carton_type value count: ', df['carton_type'].value_counts())
            print('sku_type value count: ', df['sku_type'].value_counts())

            '''箱型分布'''
            carton_index = ['month', 'weekday', 'date', 'delivery_mode', 'deliveryNO' ]
            carton_type_pivot = pd.pivot_table(df, index=carton_index,
                                               values=['vol', 'cartonNO', 'quantity'],
                                               columns=['carton_type'],
                                               aggfunc={'vol': np.sum, 'cartonNO': pd.Series.nunique, 'quantity': np.sum},
                                               fill_value=0).reset_index()

            print('箱型分布: carton_type_pivot ', carton_type_pivot)

            carton_type = ['单箱单件', '单箱单品', '单箱多品', '异常']

            ### 多级索引转成单层索引
            carton_type_col = []
            for (s1, s2) in carton_type_pivot.columns:
                if len(s2) > 0:
                    carton_type_col.append(s1 + '_' + str(s2))
                else:
                    carton_type_col.append(s1)
            carton_type_pivot.columns = carton_type_col

            ## 增加 均箱体积
            for t in carton_type[:3]:
                carton_type_pivot['均箱体积_{}'.format(t)] = carton_type_pivot['vol_{}'.format(t)] / carton_type_pivot['cartonNO_{}'.format(t)]
                carton_type_pivot['均箱体积_{}'.format(t)] = carton_type_pivot['均箱体积_{}'.format(t)].fillna(0)

            # 总体积及总箱数
            carton_type_pivot['总体积'] = sum([carton_type_pivot['vol_{}'.format(t)] for t in carton_type])
            carton_type_pivot['总箱数'] = sum([carton_type_pivot['cartonNO_{}'.format(t)] for t in carton_type])
            carton_type_pivot['总均箱体积'] = carton_type_pivot['总体积'] / carton_type_pivot['总箱数']

            print('箱型分布')
            print('*' * 20, 'carton_type_pivot')
            print(carton_type_pivot.shape)
            print(carton_type_pivot.head(20))
            carton_type_pivot = carton_type_pivot.sort_values(by=['delivery_mode', 'date'], ignore_index=True)



            '''箱内SKU分布'''
            sku_index = ['weekday', 'date', 'delivery_mode', 'deliveryNO']
            sku_type_pivot = pd.pivot_table(df, index=sku_index,
                                            values=['cartonNO'],
                                            columns=['sku_type'],
                                            aggfunc={'cartonNO': pd.Series.nunique},
                                            fill_value=0).reset_index()
            ### 多级索引转成单层索引
            sku_type_col = []
            for (s1, s2) in sku_type_pivot.columns:
                if len(s2) > 0:
                    sku_type_col.append(s1 + '_' + str(s2))
                else:
                    sku_type_col.append(s1)
            sku_type_pivot.columns = sku_type_col

            pt_col = sku_type_pivot.columns[len(sku_index):]
            # 增加总箱数
            sku_type_pivot['总箱数'] = sku_type_pivot[pt_col].sum(axis=1)

            sku_type_pivot = sku_type_pivot.sort_values(by=['delivery_mode', 'date'], ignore_index=True)

            # print('*' * 20, 'sku_type_pivot')
            # print(sku_type_pivot.dtypes)
            # print(sku_type_pivot.shape)
            # print(sku_type_pivot.head(10))

            '''日来柜数量'''
            daily_container_df = pd.pivot_table(df, index=['receive_date'],
                                                values=['containerNO', 'deliveryNO', 'trackingNO'],
                                                aggfunc=pd.Series.nunique,
                                                fill_value=0).reset_index()

            ### 去掉海柜号或跟踪号为空的计数
            daily_container_df['trackingNO'] = daily_container_df['trackingNO'] - 1
            daily_container_df['containerNO'] = daily_container_df['deliveryNO'] - daily_container_df['trackingNO']

            ### 重排列
            daily_container_df = daily_container_df[['receive_date', 'containerNO', 'trackingNO', 'deliveryNO']]

            daily_container_df['receive_interval_days'] = pd.to_timedelta(daily_container_df['receive_date'] - daily_container_df['receive_date'].shift(1)).dt.days

            daily_container_df = daily_container_df.fillna(0)

            print('*'*20, 'daily_container_df')
            print(daily_container_df.dtypes)
            print(daily_container_df.shape)
            print(daily_container_df.head(10))


            '''日来柜中海柜中的SKU数'''

            sku_index = ['month', 'weekday', 'date', 'delivery_mode', 'deliveryNO']
            daily_container_sku_df = pd.pivot_table(df, index=sku_index,
                                                values=['sku'],
                                                aggfunc={'sku': pd.Series.nunique},
                                                fill_value=0).reset_index()

            print('*'*20, "daily_container_sku_df")
            print(daily_container_sku_df)

            new_shape = df.shape
            days = df['date'].nunique()
            start_date = df['date'].min()
            end_date = df['date'].max()

            self.ib_info = '''数据源\n 物理仓: {}, 原始数据: 行数 {}, 列数 {};\n 分析天数: {}, 开始日期: {}， 结束日期: {};\n 分析数量: 行数 {}, 列数{}'''.format(
                self.ib_df['wh_code'].unique(), self.ib_df.shape[0], self.ib_df.shape[1], days, start_date, end_date, new_shape[0], new_shape[1])

            print('inbound_info: ', self.ib_info)

            self.ib_container_carton_type = carton_type_pivot
            self.ib_container_skuNum = sku_type_pivot
            self.ib_daily_container_num = daily_container_df
            self.ib_daily_container_sku = daily_container_sku_df


            self.ib_resultsTableView.setModel(dfModel(self.ib_container_carton_type))
            self.show_info_dialog('【箱数分布】计算完成!')

        except:
            self.show_error_dialog('【箱数分布】计算错误,请确认字段列编号是否正确? 分析时间段与原始数据是否有重叠!')

    def inbound_download_all(self):
        try:
            filePath, ok2 = QFileDialog.getSaveFileName(None, caption='保存文件', filter='Excel files(*.xlsx , *.xls)')

            if 'xls' in filePath or 'xlsx' in filePath:
                ### write to file
                writer = pd.ExcelWriter(filePath)
                if self.ib_date_distribution.shape[0]>0:
                    self.format_data(writer=writer, df=self.ib_date_distribution, sheet_name='C1-入库量分布', source_data_info=self.ib_info)
                if self.ib_container_carton_type.shape[0] > 0:
                    self.format_data(writer=writer, df=self.ib_container_carton_type, sheet_name='C2-箱型分布', source_data_info=self.ib_info)
                if self.ib_container_skuNum.shape[0] > 0:
                    self.format_data(writer=writer, df=self.ib_container_skuNum, sheet_name='C3-单箱SKU分布', source_data_info=self.ib_info)
                if self.ib_daily_container_num.shape[0] > 0:
                    self.format_data(writer=writer, df=self.ib_daily_container_num, sheet_name='C4-日来柜数量', source_data_info=self.ib_info)
                if self.ib_daily_container_sku.shape[0]>0:
                    self.format_data(writer=writer, df=self.ib_daily_container_sku, sheet_name='C5-日来柜SKU数', source_data_info=self.ib_info)

                writer.save()
                self.show_info_dialog('入库分析结果保存成功！')
            else:
                self.show_info_dialog('请保存成指定的文件类型！')
        except:
            self.show_error_dialog('文件保存失败！')

    def inbound_clear_all(self):
        try:
            print('In inbound_clear_all function!!!')
            self.ib_dataTableView.setModel(dfModel(pd.DataFrame()))
            self.ib_resultsTableView.setModel(dfModel(pd.DataFrame()))

            self.ib_totalRow.clear()  # 原始数据总行数
            self.ib_totalCol.clear()  # 原始数据总列数
            self.ib_errorRow.clear()  # 异常数据
            self.ib_validRow.clear()  # 有效数据总行数
            self.ib_totalSKU.clear()  # 总sku数
            self.ib_totalDays.clear()  # 总天数

        except:
            self.show_error_dialog('清理数据失败！')

    def format_data(self, writer, df, sheet_name, source_data_info=None, isTrans=True):
        '''
        将Dataframe 格式化写入Excel表格
        :param writer: Excel文件
        :param df: 写入的Dataframe
        :param sheet_name: 表格sheet名，需要根据sheet名修改格式
        :param source_data_info: 原始数据信息
        :return: None
        '''
        print('in format_data function sheet_name: ', sheet_name)
        workbook = writer.book

        '''设置格式'''
        ## 数据格式
        percent_fmt = workbook.add_format({'num_format': '0.00%'})
        pure_percent_fmt = workbook.add_format({'num_format': '0%'})
        amt_fmt = workbook.add_format({'num_format': '#,##0'})
        dec2_fmt = workbook.add_format({'num_format': '#,##0.00'})
        dec4_fmt = workbook.add_format({'num_format': '#,##0.0000'})
        date_fmt = workbook.add_format({'font_name': 'Microsoft YaHei', 'font_size': 9, 'num_format': 'yyyy/mm/dd'})

        ## 列格式
        fmt = workbook.add_format({'font_name': 'Microsoft YaHei', 'font_size': 9, 'align': 'center', 'valign': 'vcenter'})

        ## 边框格式
        border_format = workbook.add_format({'border': 1})

        ## 表头及字段格式
        head_note_fmt = workbook.add_format(
            {'bold': True, 'font_size': 11, 'font_name': 'Microsoft YaHei', 'bg_color': '2F75B5', 'font_color': 'white', 'align': 'center', 'valign': 'vcenter'})
        ## 表头逻辑说明格式
        remark_fmt = workbook.add_format(
            {'bold': False, 'font_size': 9, 'font_name': 'Microsoft YaHei', 'bg_color': '#BFBFBF', 'align': 'left', 'valign': 'vcenter'})
        # 'bg_color': '#BFBFBF','bold': True,
        note_fmt = workbook.add_format(
            {'bold': True, 'font_size': 9, 'font_name': 'Microsoft YaHei', 'bg_color': '07387D', 'font_color': 'white', 'align': 'center', 'valign': 'vcenter'})
        bold_fmt = workbook.add_format({'bold': True, 'font_size': 9})
        left_fmt = workbook.add_format({'font_size': 9, 'font_name': 'Microsoft YaHei', 'align': 'left', 'valign': 'vcenter'})

        ### 修改编号，从1开始
        df.index = range(1, len(df) + 1)
        df.index.name = '序号'

        ### df写入表格， 从第3行开始写入, 第1行为逻辑说明；第2行为数据来源
        start_row = 3

        df.to_excel(excel_writer=writer, sheet_name=sheet_name, encoding='utf8',
                    startrow=start_row, startcol=0, na_rep='-', inf_rep='-')
        worksheet1 = writer.sheets[sheet_name]

        ### 数据源行数，和列数 +1表示最后一行 start_row 为前2行说明
        end_row = df.shape[0] + 1 + start_row
        cols = df.shape[1]
        ### excel中列名 A,B,C...
        cap_list = self.get_char_list(200)
        end_col = cap_list[cols]

        ### 添加边框
        worksheet1.conditional_format('A{}:{}{}'.format(start_row + 1, end_col, end_row),
                                      {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        # 'type': 'cell','criteria': '>', 'value': 0, 'format': border_format

        ### 设置列宽
        worksheet1.set_column('A:A'.format(end_col), 6, fmt)
        worksheet1.set_column('B:{}'.format(end_col), 12, fmt)
        # worksheet1.set_row(0, 100)  # 设置测试逻辑行高
        # worksheet1.set_row(1, 50)   # 设置数据来源行高

        ### 序号列格式设置
        worksheet1.write(start_row, 0, '序号', note_fmt)
        for i, index in enumerate(df.index):
            worksheet1.write(i + start_row + 1, 0, index, fmt)

        ### 根据表名，设置页面表头及说明
        if 'A1.1' in sheet_name:
            # 规划库位推荐
            worksheet1.set_column('B:B', 15, left_fmt)
            worksheet1.set_row(0, 100)
            worksheet1.set_row(1, 50)
            ### 第一行为表格说明
            remark = '''测算逻辑 \n 1. 一个sku只匹配一种库位类型；\n 2. 按库容及sku数计算需求库位，取两者较大值；\n 3. 根据库容饱和系数计算规划库位需求；\n 4. 根据不同库位类型的库容坪效系数估算面积需求。'''

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            worksheet1.merge_range('B{}:I{}'.format(start_row, start_row), '库位参数', note_fmt)
            worksheet1.merge_range('J{}:S{}'.format(start_row, start_row), '现状批次库存', note_fmt)
            worksheet1.merge_range('T{}:Y{}'.format(start_row, start_row), '规划参数', note_fmt)

            ### 序号列格式化, 数据从第3行开始写入
            worksheet1.merge_range('A{}:A{}'.format(start_row, start_row + 1), '序号', note_fmt)

            ### 有合并行的地方，添加边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, end_row),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'A1.2' in sheet_name:
            # 现状库位统计
            worksheet1.set_row(0, 50)
            worksheet1.set_row(1, 50)
            ### 第一行为表格说明
            remark = '''测算逻辑 \n 1. 当前批次库存的库位分布；\n 2. 计算不同库位类型的体积、件数、sku数、库位数量及其占比。'''
            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})


        elif 'A2' in sheet_name:
            # 客户维度体积及件数分布
            remark = '''测算逻辑 \n 1. 客户维度，产品货型的体积和件数分布,并计算客户维度的库存深度；\n 2. 根据产品货型的比例对客户定性：
            \t ①大件体积>=80%, 纯大货型；②大件体积>=60%, 大货型；③ 大件体积>=30%, 混货型；④其他， 小货型'''

            worksheet1.set_row(0, 100)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 序号列格式化, 数据从第3行开始写入
            worksheet1.merge_range('A{}:A{}'.format(start_row, start_row + 1), '序号', note_fmt)
            worksheet1.merge_range('B{}:B{}'.format(start_row, start_row + 1), '客户代码', note_fmt)

            worksheet1.merge_range('C{}:I{}'.format(start_row, start_row), '在库体积(m³)', note_fmt)
            worksheet1.merge_range('J{}:O{}'.format(start_row, start_row), '在库体积占比', note_fmt)
            worksheet1.merge_range('P{}:V{}'.format(start_row, start_row), '在库件数', note_fmt)
            worksheet1.merge_range('W{}:AB{}'.format(start_row, start_row), '在库件数占比', note_fmt)

            worksheet1.merge_range('AC{}:AC{}'.format(start_row, start_row + 1), 'sku数', note_fmt)
            worksheet1.merge_range('AD{}:AD{}'.format(start_row, start_row + 1), '库存深度\n(m³/sku)', note_fmt)
            worksheet1.merge_range('AE{}:AE{}'.format(start_row, start_row + 1), '库存深度\n(件/sku)', note_fmt)
            worksheet1.merge_range('AF{}:AF{}'.format(start_row, start_row + 1), '大件体积占比', note_fmt)
            worksheet1.merge_range('AG{}:AG{}'.format(start_row, start_row + 1), '客户类型', note_fmt)

            ### 有合并行的地方，添加边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, end_row),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'A3' in sheet_name:
            # sku库龄等级的分布
            remark = '''测算逻辑 \n 1. sku库龄等级的分布；\n 2. sku库龄取不同库位库龄的加权平均值，权重为件数比例'''

            worksheet1.set_row(0, 100)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 更新列宽
            worksheet1.set_column('A:{}'.format(end_col), 15, fmt)
            worksheet1.set_column('B:B', 15, left_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'C1' in sheet_name:
            # 海柜及快递来货体积及件数分布
            remark = '''测算逻辑 \n 1. 日维度不同货运方式到货件型分布'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 80)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            worksheet1.merge_range('A{}:A{}'.format(start_row, start_row + 1), '序号', note_fmt)
            worksheet1.merge_range('B{}:B{}'.format(start_row, start_row + 1), 'Month', note_fmt)
            worksheet1.merge_range('C{}:C{}'.format(start_row, start_row + 1), '日期', note_fmt)
            worksheet1.merge_range('D{}:D{}'.format(start_row, start_row + 1), '货运方式', note_fmt)

            worksheet1.merge_range('E{}:K{}'.format(start_row, start_row), '来货件数', note_fmt)
            worksheet1.merge_range('L{}:R{}'.format(start_row, start_row), '来货体积', note_fmt)
            worksheet1.merge_range('S{}:X{}'.format(start_row, start_row), '来货件数占比', note_fmt)
            worksheet1.merge_range('Y{}:AD{}'.format(start_row, start_row), '来货体积占比', note_fmt)

            ### 有合并行的地方，添加边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, end_row),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'C2' in sheet_name:
            # 海柜及快递来货体积及件数分布
            remark = '''测算逻辑 \n 1. 日维度不同货运方式到货箱型类别分布; \n 2. 箱型划分：
            \t ①单箱单件：箱内sku数=1，件数=1；\t ②单箱单件：箱内sku数=1，件数>1；\t ③单箱多品：箱内sku数>1，件数>1；\t ④异常：件数=0'''

            worksheet1.set_row(0, 100)  # 设置测试逻辑行高
            worksheet1.set_row(1, 80)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            worksheet1.merge_range('A{}:A{}'.format(start_row, start_row + 1), '序号', note_fmt)
            worksheet1.merge_range('B{}:B{}'.format(start_row, start_row + 1), 'Month', note_fmt)
            worksheet1.merge_range('C{}:C{}'.format(start_row, start_row + 1), 'weekday', note_fmt)
            worksheet1.merge_range('D{}:D{}'.format(start_row, start_row + 1), 'date', note_fmt)
            worksheet1.merge_range('E{}:E{}'.format(start_row, start_row + 1), '货运方式', note_fmt)
            worksheet1.merge_range('F{}:F{}'.format(start_row, start_row + 1), '海柜号或跟踪号', note_fmt)

            worksheet1.merge_range('G{}:J{}'.format(start_row, start_row), '箱数', note_fmt)
            worksheet1.merge_range('K{}:N{}'.format(start_row, start_row), '件数', note_fmt)
            worksheet1.merge_range('O{}:R{}'.format(start_row, start_row), '体积', note_fmt)
            worksheet1.merge_range('S{}:U{}'.format(start_row, start_row), '均箱体积', note_fmt)

            worksheet1.merge_range('V{}:V{}'.format(start_row, start_row + 1), '总体积', note_fmt)
            worksheet1.merge_range('W{}:W{}'.format(start_row, start_row + 1), '总箱数', note_fmt)
            worksheet1.merge_range('X{}:X{}'.format(start_row, start_row + 1), '总均箱体积', note_fmt)

            ### 有合并行的地方，添加边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, end_row),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'C3' in sheet_name:
            # 海柜及快递来货体积及件数分布
            remark = '''测算逻辑 \n 1. 日维度不同货运方式到货箱内SKU数分布；\n 2. 箱内含1sku~9sku的箱数单独统计作为列，大于10sku的箱合并统计，只展示数据中存在的列。'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 80)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})


        elif 'C4' in sheet_name:
            # 海柜及快递来货体积及件数分布
            remark = '''测算逻辑 \n 1. 日维度海柜及快递方式到货数量，以及来货频次'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 80)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'C5' in sheet_name:
            # 海柜及快递来货体积及件数分布
            remark = '''测算逻辑 \n 1. 日维度海柜及快递中SKU数量'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 80)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        # 出库表格说明
        elif 'B1.1' in sheet_name:
            remark = '''测算逻辑 \n 1. 日维度的标准订单EIQ'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B1.2' in sheet_name:
            remark = '''测算逻辑 \n 1. 日维度的FBA订单EIQ'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B1.3' in sheet_name:
            # 订单结构EIQ
            remark = '''测算逻辑 \n 1. 日维度的订单结构EIQ; \n 2. 不同订单结构订单数,sku数,行数及件数占比'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B1.4' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 根据截单时间点划分为2个波次，开始作业时间（如7点）到截单时间前为第二波，其他为第一波; \n 2. 不同波次不同订单结构中订单数,sku数,行数及件数分布'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B1.5' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 统计历史数据中每日拣货单数量，汇总拣货单中库位数及sku数; \n 2. 历史数据中订单维度的EIQ及拣货维度EIQ'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'B2' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 统计多品多件订单货型组合分布; \n 2. 将sku根据货型尺寸定性分为4类,即小-中-大-XL, 划分依据为：①小：XS, S; ②中：M; ③大：L1, L2; ④XL
            3. 订单的货型组合定义：①小: 订单中sku货型全部为小; ②中: 订单中sku货型全部为中; ③大: 订单中sku货型全部为大; 
            ④中配小: 订单中同时存在中、小货型，且不含大货型; ⑤大配小：订单中同时存在大、小货型，且不含中货型; ⑥大配中：订单中同时存在大、中货型，且不含小货型
            ⑦大中小: 订单中同时存在sku为大、中、小货型; ⑧XL: 只要订单含有XL货型sku'''

            worksheet1.set_row(0, 100)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B3' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 客户维度库存、出库特性; \n 2. 若出库总天数≤30,则月动销sku选取所有数据计算，否则选取数据中天数最长的自然月份; \n 3. 由于库存数据选取的静态日期，月动销sku数有可能大于库存sku数'''

            worksheet1.set_row(0, 70)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 序号列格式化, 数据从第3行开始写入
            worksheet1.merge_range('A{}:A{}'.format(start_row, start_row + 1), '序号', note_fmt)
            worksheet1.merge_range('B{}:B{}'.format(start_row, start_row + 1), '客户代码', note_fmt)

            worksheet1.merge_range('C{}:G{}'.format(start_row, start_row), '库存结构', note_fmt)
            worksheet1.merge_range('H{}:N{}'.format(start_row, start_row), '出库结构', note_fmt)
            worksheet1.merge_range('O{}:S{}'.format(start_row, start_row), '出库特征', note_fmt)

            ### 有合并行的地方，添加边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, end_row),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B4' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 周维度sku动销率; \n 2. 关键字段计算逻辑：① 重合sku=current和next的交集; \t ② 流入sku=next-current的差集 \t ③ 流出sku=current-next的差集 
            ④ current重合sku件数：重合sku在current中的件数; \t ⑤ next重合sku件数：重合sku在next中的件数; ⑥ sku池变化率=流入sku/current'''

            worksheet1.set_row(0, 100)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, 2),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'B5' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 操作日期维度24小时订单流入统计; \n 2. 统计每个时点流入的订单数、行数、sku数、件数; \n 3. 统计截止到当前时刻的累计订单数、行数、sku数、件数及其比例'''

            worksheet1.set_row(0, 70)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}2'.format(end_col),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'B6.1' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. sku多重ABC分类的库存、出库分布
           '''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            worksheet1.conditional_format('A1:{}{}'.format(end_col, 2),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B6.2' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. sku多重ABC分类结果明细; \n 2. skuABC分类逻辑：\n ①出库件数ABC：A、B、C类累计出库件数分别为70%，90%，100%; \n ②出库频次ABC：出库频率=出库天数/总出库天数 A≥50%, C≤20%，B其他\n ③动碰最大间隔天数ABC： A≤3, C≥10, B其他
            3. 组合ABC逻辑: ① A: A的数量≥1,且c的数量=0 ② C: A的数量=0,且c的数量≥2 ③ B: 其他'''

            worksheet1.set_row(0, 100)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, 2),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B7.1' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. sku出库库龄等级分布,以订单行的最小维度统计; \n 2. sku出库库龄为出库明细行中出库日期与上架日期的间隔天数'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, 2),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        elif 'B7.2' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 订单出库库龄等级分布,以订单的最小维度统计; \n 2. 订单出库库龄记为订单内sku的最长库龄'''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, 2),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B8.1' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 出库渠道维度的订单分布
              '''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, 2),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})
        elif 'B8.2' in sheet_name:
            # 波次EIQ
            remark = '''测算逻辑 \n 1. 订单下单平台维度的订单分布
              '''

            worksheet1.set_row(0, 50)  # 设置测试逻辑行高
            worksheet1.set_row(1, 50)  # 设置数据来源行高

            worksheet1.merge_range('A1:{}1'.format(end_col), remark, remark_fmt)
            worksheet1.merge_range('A2:{}2'.format(end_col), source_data_info, remark_fmt)

            ### 没有有合并行的地方，添加说明行边框
            worksheet1.conditional_format('A1:{}{}'.format(end_col, 2),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': border_format})

        else:
            worksheet1.write(start_row, 0, '序号', note_fmt)

        ### 按列名设置列的格式
        for k, col in enumerate(df.columns.values):
            i = k + 1
            # 将dataframe 列名写入sheet， （行，列，列名，格式）
            worksheet1.write(start_row, i, col, note_fmt)

            ### 根据列名，格式化一列的格式
            if '%' in col or 'freq' in col or '率' in col or '占比' in col:
                # print(col, '百分数')
                worksheet1.conditional_format('{}1:{}{}'.format(cap_list[i], cap_list[i], end_row),
                                              {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': percent_fmt})
            elif 'm³' in col or '系数' in col or 'vol' in col or '体积' in col or '在库托数' in col:
                # print(col, '2位小数，千分位')
                worksheet1.conditional_format('{}1:{}{}'.format(cap_list[i], cap_list[i], end_row),
                                              {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': dec2_fmt})
            elif 'EN' in col or 'EQ' in col or 'IK' in col or 'IQ' in col or '/' in col:
                # print(col, '2位小数，千分位')
                worksheet1.conditional_format('{}1:{}{}'.format(cap_list[i], cap_list[i], end_row),
                                              {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': dec2_fmt})
            elif '日期' in col or 'date' in col:
                # print(col, '4位小数，千分位')
                worksheet1.conditional_format('{}1:{}{}'.format(cap_list[i], cap_list[i], end_row),
                                              {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': date_fmt})
            elif '库容利用率' in col:
                # print(col, '4位小数，千分位')
                worksheet1.conditional_format('{}1:{}{}'.format(cap_list[i], cap_list[i], end_row),
                                              {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': pure_percent_fmt})
            elif '等级' in col:
                # print(col, '4位小数，千分位')
                worksheet1.conditional_format('{}1:{}{}'.format(cap_list[i], cap_list[i], end_row),
                                              {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': left_fmt})
            else:
                # print(sheet_name, col, '2位小数，千分位')
                worksheet1.conditional_format('{}1:{}{}'.format(cap_list[i], cap_list[i], end_row),
                                              {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': amt_fmt})

        ### 最后一行即合计行 加粗
        if 'A1' in sheet_name:
            worksheet1.conditional_format('B{}:{}{}'.format(end_row, end_col, end_row),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': bold_fmt})
        ### 客户分布加粗总计行
        elif 'A2' in sheet_name:
            worksheet1.conditional_format('B{}:{}{}'.format(5, end_col, 5),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': bold_fmt})

        elif 'A3' in sheet_name:
            worksheet1.conditional_format('B{}:{}{}'.format(end_row, end_col, end_row),
                                          {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': bold_fmt})

        ### 是否翻译列名
        if isTrans:
            cols_name = self.trans(df.columns)
            for i, col in enumerate(cols_name):
                # 添加了index列在Excel的第1列，df的columns像右移1行
                worksheet1.write(start_row, i + 1, col, note_fmt)


    def get_char_list(self, n):
        char_list = [chr(i) for i in range(65, 91)]

        for i in range(65, 91):
            for j in range(65, 91):
                char_list.append(chr(i) + chr(j))
                if len(char_list) >= n:
                    break
            if len(char_list) >= n:
                break
        # print(char_list)
        return char_list

    def trans(self, columns):
        if len(columns) > 0:
            new_col = []
            for col in columns:
                ### 库存分析修改列名
                col = col.replace('在库体积(m³)_', '')
                col = col.replace('在库体积(M3)_', '')
                col = col.replace('在库体积(M³)_', '')
                col = col.replace('在库件数_', '')

                ### 入库分析字段翻译
                col = col.replace('receive_', '来货')
                col = col.replace('interval_', '间隔')
                col = col.replace('days', '天数')
                col = col.replace('date', '日期')

                col = col.replace('deliveryNO', '海柜号或跟踪号')
                col = col.replace('containerNO', '海柜号')
                col = col.replace('trackingNO', '跟踪号')

                col = col.replace('delivery_mode', '货运方式')

                col = col.replace('vol_', '')
                col = col.replace('quantity_', '')
                col = col.replace('cartonNO_', '')
                col = col.replace('均箱体积_', '')

                col = col.replace('cartonNO', '箱数')
                col = col.replace('vol', '体积')
                col = col.replace('quantity', '件数')

                ### 出库分析字段翻译
                col = col.replace('order_tag', '订单类型')
                col = col.replace('re_order_structure', '订单结构')
                col = col.replace('order_size_type', '订单货型组合')
                col = col.replace('customer', '客户代码')
                col = col.replace('combine_', '组合')
                col = col.replace('age_class', '库龄等级')
                col = col.replace('age', '库龄')
                col = col.replace('ob_', '出库')
                col = col.replace('cumu_', '累计')
                # col = col.replace('qty', '件数')
                col = col.replace('freq_day', '天数频率')
                col = col.replace('max_', '最大')
                col = col.replace('ob_', '出库')
                col = col.replace('inv_', '在库')
                col = col.replace('orderID', '订单数')
                col = col.replace('order_', '订单')

                col = col.replace('channel', '渠道')
                col = col.replace('platform', '平台')

                new_col.append(col)
        else:
            new_col = columns

        return new_col

class dfModel(QAbstractTableModel):

    def __init__(self, data, showAllColumn=True):
        QAbstractTableModel.__init__(self)
        self.showAllColumn = showAllColumn
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if type(self._data.columns[col]) == tuple:
                return self._data.columns[col][-1]
            else:
                return self._data.columns[col]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return (self._data.axes[0][col])
        return None



if __name__ == "__main__":

    app = QApplication(sys.argv)
    ui = MainWindow()
    # icon = ui.resource_path(os.path.join('images', 'goodcang.ico'))
    # icon = ui.resource_path(os.path.join('images', 'GC_DataAnalysis.png'))
    app.setWindowIcon(QIcon(':/GC_DataAnalysis.ico'))
    # app.setWindowIcon(QIcon('GC_DataAnalysis.ico'))
    ui.show()
    sys.exit(app.exec_())