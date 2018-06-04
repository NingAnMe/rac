# coding: UTF-8
'''
Created on 2016年7月3日

@author: taoqu_000
'''
import datetime
import os, sys, time
from configobj import ConfigObj
from netCDF4 import Dataset, stringtoarr
from PB.CSC.pb_csc_console import LogServer
import numpy as np
from multiprocessing import Pool


def run(pair, nc_type):
    part1 = pair.split('_')[0]
    if 'FY3' in part1:
        run_FY3X_LEO(pair, nc_type)


def run_FY3X_LEO(satPair, nc_type):
    # 解析配置文件
    nc_root = GbalCfg['PATH']['OUT']['ISN']

    ncPath = os.path.join(nc_root, satPair)
    ncList = [e for e in os.listdir(ncPath) if "SATCAL+%s+LEOLEOIR" % nc_type in e]

    if len(ncList) == 0: return
    ncList.sort()

    ymd1 = ncList[0].split('_')[4][:8]
    ymd2 = ncList[-1].split('_')[4][:8]

    Log.info(u'[%s] [%s-%s]' % (satPair, ymd1, ymd2))

    nc = GSICS_STD_NC(ymd1, ymd2, satPair, nc_root, nc_type)
    nc.date = []
    nc.validity_period = []
    nc.number_of_collocations = []
    nc.slope = []
    nc.offset = []

    nc.Tb_bias = []
    nc.slope_se = []
    nc.offset_se = []
    nc.std_scene_tb_bias_se = []
    nc.covariance = []

    nc.cal_slope = []
    nc.cal_offset = []
#     nc.cal_lut = []
    for eachNC in ncList:
#         ymd = eachNC.split('_')[4][:8]
        rootgrp = Dataset(os.path.join(ncPath, eachNC), 'r', format='NETCDF3_CLASSIC')
        nc.date.append(rootgrp.variables['date'][0])
        nc.validity_period.append(rootgrp.variables['validity_period'][0])
        nc.number_of_collocations.append(rootgrp.variables['number_of_collocations'][0])
        nc.slope.append(rootgrp.variables['Rad_corrct_slope'][0])
        nc.offset.append(rootgrp.variables['Rad_corrct_offset'][0])

        if nc.std_scene_tb is None:
            nc.std_scene_tb = rootgrp.variables['std_scene_tb'][:]
        nc.Tb_bias.append(rootgrp.variables['std_scene_tb_bias'][0])
        nc.slope_se.append(rootgrp.variables['slope_se'][0])
        nc.offset_se.append(rootgrp.variables['offset_se'][0])
        nc.std_scene_tb_bias_se.append(rootgrp.variables['std_scene_tb_bias_se'][0])
        nc.covariance.append(rootgrp.variables['covariance'][0])

        nc.cal_slope.append(rootgrp.variables['CAL_slope'][0])
        nc.cal_offset.append(rootgrp.variables['CAL_offset'][0])
#         nc.cal_lut.append(rootgrp.variables['CAL_LUT'][:])
        rootgrp.close()

    nc.create()
    nc.write_Global_Attr()
    nc.add_Dimensions()
    nc.add_Variables()

    nc.write_Variables_Value()
    nc.close()

#     dst_dir = '/content/thredds/public/products/cma/rac/%s-vissr/metopa-iasi/demo/' % sat1.lower()
#     dst_fp = os.path.join(dst_dir, nc.nc_name)
#     pair_lst.append((nc.nc_path, dst_fp))

    Log.info(u'Success')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def getTime_Now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


FV = -999


class GSICS_STD_NC(object):

    def __init__(self, ymd1, ymd2, satPair, output_root_dir, nc_type):

        self.satPair = satPair
        part1, part2 = satPair.split('_')
        self.sat1 , self.sen1 = part1.split("+")
        self.sat2 , self.sen2 = part2.split("+")
        self.nc_type = nc_type.upper()

        if self.sat1.startswith("FY2"): self.sen1 = "SVISSR"  # VISSR -> SVISSR
        if self.sat2.startswith("METOP"): self.sat2 = self.sat2.replace("METOP-", "MetOp")  # METOP -> MetOp

        self.ymd1 = ymd1
        self.ymd2 = ymd2
        self.hms = "000000"

#         self.y_m_d = "%s-%s-%s" % (ymd[:4], ymd[4:6], ymd[6:])

        self.nc_dir = output_root_dir
        self.nc_name = 'W_CN-CMA-NSMC,SATCAL+%s+LEOLEOIR,%s+%s-%s+%s_C_BABJ_%s%s_demo_01.nc' % \
                        (nc_type, self.sat1, self.sen1, self.sat2, self.sen2, self.ymd1, self.hms)
        self.nc_path = os.path.join(self.nc_dir, self.nc_name)

        attr_fname = '%sX_%s.attr' % (self.sat1[:3], nc_type)
        self.conf = ConfigObj(os.path.join(MAIN_PATH, attr_fname))
        self.chan = int(self.conf['%s+%s' % (self.sat1, self.sen1)]['_chan'])
        self.chanlist = self.conf['%s+%s' % (self.sat1, self.sen1)]['_chanlist']
        self.Custom_Global_Attrs = {}
        self.set_Custom_Global_Attrs()
        self.set_history()

        self.std_scene_tb = None
        #
        self.date = None
        self.validity_period = None
        self.number_of_collocations = None

        self.Tb_bias = [FV] * self.chan
        self.slope_se = [FV] * self.chan
        self.offset_se = [FV] * self.chan
        self.std_scene_tb_bias_se = [FV] * self.chan
        self.covariance = [FV] * self.chan

        self.cal_slope = None
        self.cal_offset = None
#         self.cal_lut = None

    def set_custom_value(self, std_scene_tb, Tb_bias_lst, slope, slope_se,
                         offset, offset_se, std_scene_tb_bias_se, covariance):
        self.slope = slope
        self.offset = offset
        self.std_scene_tb = std_scene_tb
        self.Tb_bias = Tb_bias_lst
        self.slope_se = slope_se
        self.offset_se = offset_se
        self.std_scene_tb_bias_se = std_scene_tb_bias_se
        self.covariance = covariance

    def create(self):
        if not os.path.isdir(self.nc_dir):
            os.makedirs(self.nc_dir)
        self.rootgrp = Dataset(self.nc_path, 'w', format='NETCDF3_CLASSIC')  # why not use "NETCDF4"

    def set_history(self):
        '''
        历史属性设定，提出来方便修改
        '''
        self.Custom_Global_Attrs["history"] = \
        """none"""

    def set_Custom_Global_Attrs(self):
        '''
        非固定不变的属性
        '''
        self.Custom_Global_Attrs["title"] = \
        "GSICS %s+%s vs %s+%s GSICS Near Real-Time Correction" % \
        (self.sat1, self.sen1, self.sat2, self.sen2)

        self.Custom_Global_Attrs["keywords"] = \
        "GSICS, satellite, remote sensing, inter-calibration, near real time correction, " + \
        "GEO-LEO-IR, %s+%s, %s+%s, infrared" % \
        (self.sat1, self.sen1, self.sat2, self.sen2)

        self.Custom_Global_Attrs["monitored_instrument"] = "%s %s" % (self.sat1, "S-VISSR")  # zt modified
        self.Custom_Global_Attrs["reference_instrument"] = "%s %s" % (self.sat2, self.sen2)

        str_now = getTime_Now()
        self.Custom_Global_Attrs["date_created"] = str_now
        self.Custom_Global_Attrs["date_modified"] = str_now

        self.Custom_Global_Attrs["id"] = self.nc_name

        # "time_coverage" same as "validity_period"
        date_1 = datetime.date(int(self.ymd1[0:4]), int(self.ymd1[4:6]), int(self.ymd1[6:8])) + \
                datetime.timedelta(days=-4)
        date_2 = datetime.date(int(self.ymd2[0:4]), int(self.ymd2[4:6]), int(self.ymd2[6:8]))
        self.Custom_Global_Attrs["time_coverage_start"] = '%sT00:00:00Z' % date_1.strftime('%Y-%m-%d')
        self.Custom_Global_Attrs["time_coverage_end"] = '%sT23:59:59Z' % date_2.strftime('%Y-%m-%d')

    def write_Global_Attr(self):
        '''
        写全局属性
        '''
        attrs = self.conf['GLOBAL']
        for eachkey in attrs:
            if attrs[eachkey] == 'DUMMY':
                attrs[eachkey] = self.Custom_Global_Attrs[eachkey]
            if is_number(attrs[eachkey]):
                if '.' in attrs[eachkey]:
                    self.rootgrp.setncattr(eachkey, np.float32(attrs[eachkey]))
                else:
                    self.rootgrp.setncattr(eachkey, np.short(attrs[eachkey]))
            else:
                self.rootgrp.setncattr(eachkey, attrs[eachkey])

    def add_Dimensions(self):
        var_lst = self.conf['%s+%s' % (self.sat1, self.sen1)]
        self.rootgrp.createDimension('chan', int(var_lst['_chan']))  # band num
        self.rootgrp.createDimension('chan_strlen', int(var_lst['_chan_strlen']))  # band name length
        self.rootgrp.createDimension('date', None)  # None: UNLIMITED
        self.rootgrp.createDimension('validity', 2)
#         self.rootgrp.createDimension('lut_row', int(var_lst['_lut_row']))

    def add_Variables(self):
        '''
        根据配置文件内容添加数据集
        '''
        var_dict = self.conf['%s+%s' % (self.sat1, self.sen1)]
        # add CAL_LUT for each band
        dsetNameLst = var_dict.keys()
#         for eachchan in var_dict["_chanlist"]:
#             dsetNameLst.append("CAL_LUT_CH%s" % eachchan)
        for eachVar in dsetNameLst:
            if eachVar.startswith('_'): continue
            if eachVar == 'TBB_Corrct_LUT': continue
            if eachVar == 'Nonlinear_coefficient': continue
#             if eachVar.startswith('CAL_LUT'):
#                 var_info = var_dict["CAL_LUT"]
#                 var_info['_dims'] = ['date', 'lut_row']
#             else:
#                 var_info = var_dict[eachVar]
            var_info = var_dict[eachVar]
            var = self.rootgrp.createVariable(eachVar, var_info['_fmt'], var_info['_dims'])
            for eachKey in var_info:
                if eachKey.startswith('_'): continue
                if eachKey == eachVar:
                    if var_info['_fmt'] == 'S1':
                        # 字符串
                        # 需要将字符串用stringtoarr转成char的数组，再写入NC !!!
                        char_len = 1
                        for each in var_info['_dims']:
                            char_len = char_len * int(var_dict['_%s' % each])  # 计算字符总个数
                        char_ary = stringtoarr(''.join(var_info[eachKey]), char_len)
                        var[:] = char_ary
                    else:
                        # 非字符串
                        var[:] = var_info[eachKey]
                else:
                    if is_number(var_info[eachKey]):
                        if '.' in var_info[eachKey]:
                            var.setncattr(eachKey, np.float32(var_info[eachKey]))
                        else:
                            var.setncattr(eachKey, np.short(var_info[eachKey]))
                    else:
                        var.setncattr(eachKey, var_info[eachKey])

    def write_central_wavelength(self):
        '''
        写数据集内容，这些内容是非固定不变的
        '''
        # central_wavelength
        wnc = self.conf['%s+%s' % (self.sat1, self.sen1)]['wnc']['wnc']
        wnc = [float(each) for each in wnc]

        c_w = np.divide([1.] * self.chan, wnc) * 0.01
#         c_w = ['%.2e' % each for each in c_w]
        self.rootgrp.variables['central_wavelength'][:] = c_w

    def write_Variables_Value(self):

        self.write_central_wavelength()
        # date
        self.rootgrp.variables['date'][:] = self.date

        # validity_period
        self.rootgrp.variables['validity_period'][:] = self.validity_period

        # slope
        self.rootgrp.variables['Rad_corrct_slope'][:] = self.slope
        # slope_se
        self.rootgrp.variables['slope_se'][:] = self.slope_se
        # offset
        self.rootgrp.variables['Rad_corrct_offset'][:] = self.offset
        # offset_se
        self.rootgrp.variables['offset_se'][:] = self.offset_se
        # covariance
        self.rootgrp.variables['covariance'][:] = self.covariance

        # std_scene_tb
        self.rootgrp.variables['std_scene_tb'][:] = self.std_scene_tb
        # std_scene_tb_bias
        self.rootgrp.variables['std_scene_tb_bias'][:] = self.Tb_bias
        # std_scene_tb_bias_se
        self.rootgrp.variables['std_scene_tb_bias_se'][:] = self.std_scene_tb_bias_se
        # number_of_collocations
        self.rootgrp.variables['number_of_collocations'][:] = self.number_of_collocations

        # cal_slope cal_offset
        self.rootgrp.variables['CAL_slope'][:] = self.cal_slope
        self.rootgrp.variables['CAL_offset'][:] = self.cal_offset

#         cal_lut = np.array(self.cal_lut)  # 3d list -> ndarray
#
#         for i, eachchan in enumerate(self.chanlist):
#             self.rootgrp.variables['CAL_LUT_CH%s' % eachchan][:] = cal_lut[:, :, i]

    def close(self):
        self.rootgrp.close()


######################### 程序全局入口 ##############################
if __name__ == "__main__":
    # 获取程序参数接口
    ARGS = sys.argv[1:]
    HELP_INFO = \
        u"""
        [参数1]：pair 卫星对
        [样例]: python app.py pair
        """
    if "-h" in ARGS:
        print HELP_INFO
        sys.exit(-1)

    # 获取程序所在位置，拼接配置文件
    MAIN_PATH, MAIN_FILE = os.path.split(os.path.realpath(__file__))
    PROJECT_PATH = os.path.dirname(MAIN_PATH)
    CONFIG_FILE = os.path.join(PROJECT_PATH, "cfg", "global.cfg")

    PYTHON_PATH = os.environ.get("PYTHONPATH")
    DV_PATH = os.path.join(PYTHON_PATH, "DV")

    # 配置不存在预警
    if not os.path.isfile(CONFIG_FILE):
        print (u"配置文件不存在 %s" % CONFIG_FILE)
        sys.exit(-1)

    # 载入配置文件
    GLOBAL_CONFIG = ConfigObj(CONFIG_FILE)
    PARAM_DIR = GLOBAL_CONFIG['PATH']['PARAM']
    MATCH_DIR = GLOBAL_CONFIG['PATH']['MID']['MATCH_DATA']
    LogPath = GLOBAL_CONFIG['PATH']['OUT']['LOG']
    Log = LogServer(LogPath)  # 初始化日志

    if len(ARGS) == 1:
        # 获取命令行输入的参数 配置文件和开始结束时间  global.cfg
        SAT_PAIR = ARGS[0]

        for NC_TYPE in ["NRTC", "RAC"]:
            Log.info(u'开始运行国际标准%s的合成程序-----------------------------' % NC_TYPE)
            run(SAT_PAIR, NC_TYPE)
    else:
        print HELP_INFO
        sys.exit(-1)
