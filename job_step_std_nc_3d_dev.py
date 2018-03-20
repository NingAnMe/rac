# coding: UTF-8
'''
Created on 2016年7月3日

@author: taoqu_000, anning
'''
import datetime, calendar
import math
import os, sys, time
from posixpath import join as urljoin

from configobj import ConfigObj
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset, stringtoarr
from numpy.lib.polynomial import polyfit
from numpy.ma.core import std, mean
from numpy.ma.extras import corrcoef
# from scipy.interpolate.fitpack2 import UnivariateSpline
import h5py
from PB import pb_time
from PB.CSC.pb_csc_console import LogServer
import numpy as np
from multiprocessing import Pool, Lock


def run(matching, ymd, nc_type):
    part1 = matching.split('_')[0]
    if 'FY3' in part1:
        run_FY3X_LEO(matching, ymd, nc_type)


def run_FY3X_LEO(matching, ymd, nc_type):
    Log.info(u'[%s] [%s]' % (matching, ymd))
    # 解析配置文件

    opath = GbalCfg['PATH']['OUT']['ISN']
    print('1')
    nc = GSICS_STD_NC(ymd, matching, opath, nc_type)
    print('2')
    nc.loadMatchedPointHDF5()
    print('3')
    nc.calculate_LUT()
    print('4')  # zt add 20160825
    if nc.error == False:
        print('5')
        nc.create()
        print('6', ymd)
        nc.write()
        print('7')
        nc.close()
        Log.info(u'Success')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def getTime_Now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

#==============================================================================
# 计算斜率和截距
#==============================================================================
def G_reg1d(xx, yy, ww=None):
    ''' 
    计算斜率和截距 
    ww: weights
    '''
    rtn = []
    ab = polyfit(xx, yy, 1, w=ww)
    rtn.append(ab[0])
    rtn.append(ab[1])
    rtn.append(std(yy) / std(xx))
    rtn.append(mean(yy) - rtn[2] * mean(xx))
    r = corrcoef(xx, yy)
    rr = r[0, 1] * r[0, 1]
    rtn.append(rr)
    return rtn

def cal_RAC(leoRad, fy3Rad, delta):

    x = np.array(leoRad)
    y = np.array(fy3Rad)
    dt = np.array(delta)

    a1 = np.sum(np.square(x) / np.square(dt))
    a2 = np.sum(y / np.square(dt))
    b1 = np.sum(x / np.square(dt))
    b2 = np.sum(x * y / np.square(dt))
    c1 = np.sum(1 / np.square(dt))

    d1 = (c1 * a1 - b1 * b1)

    a = (a1 * a2 - b1 * b2) / d1
    b = (c1 * b2 - b1 * a2) / d1

    offset_se = a1 / d1
    slope_se = c1 / d1
    cov_ab = -b1 / d1

    return a, b, offset_se, slope_se, cov_ab


FV = -999


class GSICS_STD_NC(object):
    def __init__(self, ymd, matching, output_root_dir, nc_type):

        self.error = False
        self.matching = matching
        part1, part2 = matching.split('_')
        self.sat1 , self.sen1 = part1.split("+")
        self.sat2 , self.sen2 = part2.split("+")
        self.nc_type = nc_type.upper()
        print('8')
        # 命名转成国际规则
        if self.sat1.startswith("FY2"): self.sen1 = "SVISSR"  # VISSR -> SVISSR
        if self.sat2.startswith("METOP"): self.sat2 = self.sat2.replace("METOP-", "MetOp")  # METOP -> MetOp
        print('9')
        self.ymd = ymd
        self.hms = "000000"
        self.y_m_d = "%s-%s-%s" % (ymd[:4], ymd[4:6], ymd[6:])
        print('10')
        self.nc_dir = os.path.join(output_root_dir, self.matching)
        self.nc_name = 'W_CN-CMA-NSMC,SATCAL+%s+LEOLEOIR,%s+%s-%s+%s_C_BABJ_%s%s_demo_01.nc' % \
                        (self.nc_type, self.sat1, self.sen1, self.sat2, self.sen2, self.ymd, self.hms)
        self.nc_path = os.path.join(self.nc_dir, self.nc_name)
        print('11')
        Tbb2Rad_filename = '%s_LUT_TB_RB.TXT' % part1
        Tbb2Rad_fp = os.path.join(PARAM_DIR, Tbb2Rad_filename)
        self.Tbb_Rad_LUT = np.loadtxt(Tbb2Rad_fp, ndmin=2)
        print('12')
        attr_fname = '%sX_%s.attr' % (self.sat1[:3], self.nc_type)
        self.conf = ConfigObj(os.path.join(MainPath, attr_fname))
        self.chanlist = self.conf['%s+%s' % (self.sat1, self.sen1)]['_chanlist']
        self.chan = int(self.conf['%s+%s' % (self.sat1, self.sen1)]['_chan'])
        self.ndays = int(self.conf['%s+%s' % (self.sat1, self.sen1)]['_ndays'])
        self.changeYMD = self.conf['%s+%s' % (self.sat1, self.sen1)]["_change"]
        print('13')
        self.Custom_Global_Attrs = {}
        self.set_Custom_Global_Attrs()
        self.set_history()
        print('14')
        # init
        self.std_scene_tb = [float(e) for e in self.conf['%s+%s' % (self.sat1, self.sen1)]['std_scene_tb']['std_scene_tb']]
        self.Tb_bias = [FV] * self.chan
        self.slope = [FV] * self.chan
        self.slope_se = [FV] * self.chan
        self.offset = [FV] * self.chan
        self.offset_se = [FV] * self.chan
        self.std_scene_tb_bias_se = [FV] * self.chan
        self.covariance = [FV] * self.chan
        self.number_of_collocations = [FV] * self.chan
        print('15')
        self.Tbb_lut = None
        self.a_lut = [FV] * self.chan
        self.b_lut = [FV] * self.chan
        self.k0k1k2 = None
        print('16')
    def _make_NRTC_dtlist(self):
        dt_lst = []
        for n in range(self.ndays):
            dt = datetime.date(int(self.ymd[0:4]), int(self.ymd[4:6]), int(self.ymd[6:8])) - \
                    datetime.timedelta(days=n)
            nc_ymd = dt.strftime("%Y%m%d")
            dt_lst.append(dt)
            if nc_ymd in self.changeYMD:
                break
        return dt_lst
        
    def _make_RAC_dtlist(self):
        
        ndays_minus = int(self.ndays / 2)
        ndays_plus = int((self.ndays - 1) / 2)
        
        dt_lst = []
        for n in range(ndays_minus + 1):
            dt = datetime.date(int(self.ymd[0:4]), int(self.ymd[4:6]), int(self.ymd[6:8])) - \
                    datetime.timedelta(days=n)
            nc_ymd = dt.strftime("%Y%m%d")
            dt_lst.append(dt)
            if nc_ymd in self.changeYMD:
                break
        for n in range(1, ndays_plus + 1):
            dt = datetime.date(int(self.ymd[0:4]), int(self.ymd[4:6]), int(self.ymd[6:8])) + \
                    datetime.timedelta(days=n)
            nc_ymd = dt.strftime("%Y%m%d")
            if nc_ymd in self.changeYMD:
                break
            dt_lst.append(dt)
        return dt_lst
        
    def loadMatchedPointHDF5(self):
        """
        读取匹配点 HDF5 文件
        :return:
        """
        print('17')
        if self.nc_type == "NRTC":
            dt_lst = self._make_NRTC_dtlist()
        else:
            dt_lst = self._make_RAC_dtlist()
        print('18')
        for i, ib in enumerate(self.chanlist):
            ib = int(ib)
            fy3 = None
            leo = None
            weight = None

            x_dn = None
            y_rad = None
            weight_dn = None

            # make dt list
            print('dt_lst', len(dt_lst), dt_lst)
            for nc_dt in dt_lst:
                nc_ymd = nc_dt.strftime("%Y%m%d")
                oname = 'COLLOC+LEOLEOIR,%s_C_BABJ_%s.hdf5' % (self.matching, nc_ymd)
                print('19')
                # read matchedpoint from NC
                matchedpoint_nc = urljoin(MATCH_DIR, self.matching, oname)
                print(matchedpoint_nc, os.path.isfile(matchedpoint_nc))
                if not os.path.isfile(matchedpoint_nc):
                    print('19.2')
#                     Log.error("no file: %s" % matchedpoint_nc)
                    continue
                try:
                    print('19.4')
                    with h5py.File(matchedpoint_nc, 'r') as hdf5File:
                        print('19.5')
                        chan = 'CH_%02d' % ib

                        channel_group = hdf5File[chan]

                        fy3Rad = channel_group.get('S1_FovRadMean', np.empty(shape=(0, 0)))[:]
                        fy3RadStd = channel_group.get('S1_FovRadStd', np.empty(shape=(0, 0)))[:]

                        fy3DN = channel_group.get('S1_FovDnMean', np.empty(shape=(0)))[:]
                        fy3DNStd = channel_group.get('S1_FovDnStd', np.empty(shape=(0)))[:]

                        leoRad = channel_group.get('S2_FovRadMean', np.empty(shape=(0)))[:]

                        FyTime = channel_group.get('S1_Time', np.empty(shape=(0)))[:]
                        print('20')
                        # FY2时间过滤: band1-3保留白天  band4保留晚上。 FY3需要吗？？？暂时不分白天晚上
                        jd = FyTime / 24. / 3600.
                        hour = ((jd - jd.astype('int8')) * 24).astype('int8')
        #                 if ib <= 3:
        #                     index = (hour < 10)  # utc hour < 10 is day
        #                 else:
        #                     index = (hour >= 10)
                        index = (hour < 24)
                        index = np.logical_and(index, fy3RadStd > 0)
                        print('21')
                        if leo is None:
                            leo = leoRad[index]
                            fy3 = fy3Rad[index]
                            weight = fy3RadStd[index]
                        else:
                            leo = np.concatenate((leo, leoRad[index]))
                            fy3 = np.concatenate((fy3, fy3Rad[index]))
                            weight = np.concatenate((weight, fy3RadStd[index]))
                        print('22')
                        if x_dn is None:
                            x_dn = fy3DN[index]
                            y_rad = leoRad[index]
                            weight_dn = fy3DNStd[index]
                        else:
                            x_dn = np.concatenate((x_dn, fy3DN[index]))
                            y_rad = np.concatenate((y_rad, leoRad[index]))
                            weight_dn = np.concatenate((weight_dn, fy3RadStd[index]))
                except IOError:
                    continue
                print('22.5')
            print('23')
            if leo is None or len(leo) == 0: 
                self.error = True
                continue
            
            # 杂点过滤
            ab = G_reg1d(fy3, leo, 1.0 / weight)
            idx = Dust_Filter(fy3, leo, ab[0], ab[1])
            fy3 = fy3[idx]
            leo = leo[idx]
            weight = weight[idx]
            x_dn = x_dn[idx]
            y_rad = y_rad[idx]
            weight_dn = weight_dn[idx]
            print('24')
            # 至少有10个匹配点才计算
            if len(leo) > 10:
                print('24.1')
                print('count:fy3, leo, weight', len(fy3), len(leo), len(weight))
                self.calculate(i, fy3, leo, weight)
                print('24.2')
            else:
                self.error = True
            print('24.5')
            if len(x_dn) > 10:
                self.calculate_ab_LUT(i, x_dn, y_rad, weight)
            print('25')
    def calculate(self, ib, fy3Rad, leoRad, weight):
        """
        ib starts from 0
        """
        print('26')
        Tb_reference = self.std_scene_tb[ib]  # 标准温度
        # 业务的辐射比对回归系数，自变量（X轴 FY3_Rad; Y轴 leoRad)
        RadCompare_coeff = G_reg1d(fy3Rad, leoRad, 1.0 / weight)
        print('27')
        # 相关性太低的结果不要输出
        if RadCompare_coeff[4] < 0.9:
            Log.info("ib: %d  rr: %f" % (ib, RadCompare_coeff[4]))
            return
        print('28')
        # Equation 7~11
        aa, bb, offset_se, slope_se, cov_ab = cal_RAC(leoRad, fy3Rad, weight)
        dta = math.sqrt(abs(offset_se))
        dtb = math.sqrt(abs(slope_se))
        print('29')
        # 标准温度 -> RAD
        LGEO = np.interp(Tb_reference, self.Tbb_Rad_LUT[:, 0], self.Tbb_Rad_LUT[:, ib + 1])
        print('30')
        # 计算 std_scene_tb_bias_se
        delta_leo = math.sqrt((dta / bb) ** 2 + ((LGEO - aa) * dtb) ** 2 - 2 * (LGEO - aa) / bb * cov_ab)  # Equation 3
#         Tb_plus_delta = UnivariateSpline(self.Tbb_Rad_LUT[:, ib + 1], self.Tbb_Rad_LUT[:, 0])(LGEO + delta_leo)  # RAD -> TB
        Tb_plus_delta = np.interp(LGEO + delta_leo, self.Tbb_Rad_LUT[:, ib + 1], self.Tbb_Rad_LUT[:, 0])
        std_scene_tb_bias_se = abs(Tb_reference - Tb_plus_delta)  # abs
        print('31')
        self.slope_se[ib] = dtb
        self.offset_se[ib] = dta
        self.std_scene_tb_bias_se[ib] = std_scene_tb_bias_se
        self.covariance[ib] = cov_ab
        print('32')
        # 业务的参考温度处的亮温偏差
        # because LGEO = b*LGEO_corrected + a, so LGEO_corrected = (LGEO - a) / b
#         LGEO_corrected = (LGEO - aa) / bb  # zhangtao change for test
        LGEO_corrected = LGEO * RadCompare_coeff[0] + RadCompare_coeff[1]
#         Tb_corrected = UnivariateSpline(self.Tbb_Rad_LUT[:, ib + 1], self.Tbb_Rad_LUT[:, 0])(LGEO_corrected)
        Tb_corrected = np.interp(LGEO_corrected, self.Tbb_Rad_LUT[:, ib + 1], self.Tbb_Rad_LUT[:, 0])
        print('33')
        Tb_bias = Tb_reference - Tb_corrected  # FY2 - 标准
        self.Tb_bias[ib] = Tb_bias
        print('34')
        self.slope[ib] = RadCompare_coeff[0]
        self.offset[ib] = RadCompare_coeff[1]
        self.number_of_collocations[ib] = fy3Rad.size
        print('35')
    def calculate_ab_LUT(self, i, x_dn, y_rad, weight):
        '''
        i starts from 0
        '''
        # X轴 FY Rad; Y轴 leoRad
        coeff = G_reg1d(x_dn, y_rad, 1.0 / weight)
        # 相关性太低的结果不要输出
        if coeff[4] < 0.9:
            return
        
        self.a_lut[i] = coeff[0]
        self.b_lut[i] = coeff[1]
        
    def calculate_LUT(self):

        lut = [self.Tbb_Rad_LUT[:, 0]]
        for ib in xrange(1, self.chan + 1):
            aa = self.slope[ib - 1]
            bb = self.offset[ib - 1]
            
            if aa == FV and bb == FV:
                tbb = np.full_like(self.Tbb_Rad_LUT[:, ib], FV)
            else:
                rad = aa * self.Tbb_Rad_LUT[:, ib] + bb
#                 tbb = UnivariateSpline(self.Tbb_Rad_LUT[:, ib], self.Tbb_Rad_LUT[:, 0])(rad)
                tbb = np.interp(rad, self.Tbb_Rad_LUT[:, ib], self.Tbb_Rad_LUT[:, 0])

#                 idx = tbb > 350
#                 tbb[idx] = 350.
#                 idx = tbb < 150
#                 tbb[idx] = 150.
            lut.append(tbb)
        
        self.Tbb_lut = np.array(lut).transpose()

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
        if self.nc_type == "NRTC":
            date_1 = datetime.date(int(self.ymd[0:4]), int(self.ymd[4:6]), int(self.ymd[6:8])) - \
                    datetime.timedelta(days=(self.ndays - 1))
            date_2 = datetime.date(int(self.ymd[0:4]), int(self.ymd[4:6]), int(self.ymd[6:8]))
        else:
            ndays_minus = int(self.ndays / 2)
            ndays_plus = int((self.ndays - 1) / 2)
            date_1 = datetime.date(int(self.ymd[0:4]), int(self.ymd[4:6]), int(self.ymd[6:8])) - \
                    datetime.timedelta(days=ndays_minus)
            date_2 = datetime.date(int(self.ymd[0:4]), int(self.ymd[4:6]), int(self.ymd[6:8])) + \
                    datetime.timedelta(days=ndays_plus)

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
        self.rootgrp.createDimension('lut_row', int(var_lst['_lut_row']))
        self.rootgrp.createDimension('chan+1', int(var_lst['_chan']) + 1)  # band num
        self.rootgrp.createDimension('k3', 3)  # 3列 非线性拟合系数： k0 k1 k2

    def add_Variables(self):
        """
        根据配置文件内容添加数据集
        """
        var_lst = self.conf['%s+%s' % (self.sat1, self.sen1)]
        for eachVar in var_lst:
            if eachVar.startswith('_'): continue
            var_info = var_lst[eachVar]
#             print eachVar
            var = self.rootgrp.createVariable(eachVar, var_info['_fmt'], var_info['_dims'])
            for eachKey in var_info:
                if eachKey.startswith('_'): continue
                if eachKey == eachVar:
                    if var_info['_fmt'] == 'S1':
                        # 字符串
                        # 需要将字符串用stringtoarr转成char的数组，再写入NC !!!
                        char_len = 1
                        for each in var_info['_dims']:
                            char_len = char_len * int(var_lst['_%s' % each])  # 计算字符总个数
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

    def write_Variables_Value(self):
        '''
        写数据集内容，这些内容是非固定不变的
        '''
        # central_wavelength
        wnc = self.conf['%s+%s' % (self.sat1, self.sen1)]['wnc']['wnc']
        wnc = [float(each) for each in wnc]

        c_w = np.divide([1.] * self.chan, wnc) * 0.01
#         c_w = ['%.2e' % each for each in c_w]
        self.rootgrp.variables['central_wavelength'][:] = c_w

        # date
        ymdhms = '%s%s' % (self.ymd, self.hms)
        d1 = time.strptime(ymdhms, '%Y%m%d%H%M%S')
        dt = calendar.timegm(d1)
        print([dt, float(ymdhms)])
        self.rootgrp.variables['date'][0] = [dt, float(ymdhms)]

        # validity_period
        d1 = time.strptime(self.Custom_Global_Attrs["time_coverage_start"], '%Y-%m-%dT%H:%M:%SZ')
        v1 = calendar.timegm(d1)

        d2 = time.strptime(self.Custom_Global_Attrs["time_coverage_end"], '%Y-%m-%dT%H:%M:%SZ')
        v2 = calendar.timegm(d2)
        self.rootgrp.variables['validity_period'][0] = [v1, v2]

        # slope
        self.rootgrp.variables['Rad_corrct_slope'][0] = self.slope
        # slope_se
        self.rootgrp.variables['slope_se'][0] = self.slope_se
        # offset
        self.rootgrp.variables['Rad_corrct_offset'][0] = self.offset
        # offset_se
        self.rootgrp.variables['offset_se'][0] = self.offset_se
        # covariance
        self.rootgrp.variables['covariance'][0] = self.covariance

        # std_scene_tb
        self.rootgrp.variables['std_scene_tb'][:] = self.std_scene_tb
        # std_scene_tb_bias
        print(self.Tb_bias)
        self.rootgrp.variables['std_scene_tb_bias'][0] = self.Tb_bias
        # std_scene_tb_bias_se
        self.rootgrp.variables['std_scene_tb_bias_se'][0] = self.std_scene_tb_bias_se
        # number_of_collocations
        self.rootgrp.variables['number_of_collocations'][0] = self.number_of_collocations
        # cal_lut
        self.rootgrp.variables['TBB_Corrct_LUT'][:] = self.Tbb_lut
        self.rootgrp.variables['CAL_slope'][0] = self.a_lut
        self.rootgrp.variables['CAL_offset'][0] = self.b_lut
        
        NonLinCoef = self.conf['%s+%s' % (self.sat1, self.sen1)]['Nonlinear_coefficient']['Nonlinear_coefficient']
        self.rootgrp.variables['Nonlinear_coefficient'][:] = np.array(NonLinCoef).reshape((self.chan, 3))
        

    def write(self):
        print('write 36')
        self.write_Global_Attr()
        print('write 37')
        self.add_Dimensions()
        print('write 38')
        self.add_Variables()
        print('write 39')
        self.write_Variables_Value()
        print(self.rootgrp.variables['date'][0])
        print(self.rootgrp.variables['std_scene_tb_bias'][0])
        print('write 40')
    def close(self):
        self.rootgrp.close()

def Dust_Filter(x, y, a, b, mul=8):
    '''
    过滤 正负 mean delta+4倍std 的杂点,返回 idx
    x,y 是 array
    a,b 是 slope, offset
    '''
    reg_line = x * a + b
    delta = np.abs(y - reg_line)
    mean_delta = np.mean(delta)
    std_delta = np.std(delta)
    max_y = reg_line + mean_delta + std_delta * mul
    min_y = reg_line - mean_delta - std_delta * mul
    
    idx = np.logical_and(y < max_y, y > min_y)
    
    return idx


# 获取程序参数接口
args = sys.argv[1:]

help_info = \
    '''
    国际标准NC日产品生成程序
    [参数样例1]：SAT1+SENSOR1_SAT2+SENSOR2  YYYYMMDD-YYYYMMDD  nrtc(or rac)
    [参数样例2]：跟1个参数(nrtc or rac)，自动处理所有pair
    '''
if '-h' in args:
    print help_info
    sys.exit(-1)

# 获取程序所在位置，拼接配置文件
MainPath, MainFile = os.path.split(os.path.realpath(__file__))
ProjPath = os.path.dirname(MainPath)
cfgFile = os.path.join(ProjPath, 'cfg', 'global.cfg')

# 配置不存在预警
if not os.path.isfile(cfgFile):
    print (u'配置文件不存在 %s' % cfgFile)
    sys.exit(-1)

# 载入配置文件
GbalCfg = ConfigObj(cfgFile)
PARAM_DIR = GbalCfg['PATH']['PARAM']
MATCH_DIR = GbalCfg['PATH']['MID']['MATCH_DATA']
LogPath = GbalCfg['PATH']['OUT']['LOG']
Log = LogServer(LogPath)  # 初始化日志

# 开启进程池
threadNum = int(GbalCfg['CROND']['threads'])
pool = Pool(processes=threadNum)

if len(args) == 3:
    satPair = args[0]
    str_time = args[1]
    nc_type = args[2].upper()
    if nc_type not in ["NRTC", "RAC"]:
        print "arg 3 must be nrtc or rac !"
        sys.exit(-1)
    
    Log.info(u'手动%s产品生成程序开始运行-----------------------------' % nc_type)

    date_s, date_e = pb_time.arg_str2date(str_time)

    while date_s <= date_e:
        ymd = date_s.strftime('%Y%m%d')
        if threadNum == 1:
            run(satPair, ymd, nc_type)
        else:
            pool.apply_async(run, (satPair, ymd, nc_type))
        date_s = date_s + relativedelta(days=1)
    pool.close()
    pool.join()

elif len(args) == 1:
    nc_type = args[0].upper()
    if nc_type not in ["NRTC", "RAC"]:
        print "arg 1 must be nrtc or rac !"
        sys.exit(-1)
        
    Log.info(u'自动%s产品生成程序开始运行 -----------------------------' % nc_type)
    rolldays = GbalCfg['CROND']['rolldays']
    pairLst = GbalCfg['PAIRS'].keys()
    for satPair in pairLst:
        for rdays in rolldays:
            ymd = (datetime.datetime.utcnow() - relativedelta(days=int(rdays))).strftime('%Y%m%d')
            pool.apply_async(run, (satPair, ymd, nc_type))
    pool.close()
    pool.join()
else:
    print help_info
    sys.exit(-1)

