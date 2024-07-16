"""
# -*- utf-8 coding -*-#
作者：沐枫
日期：2021年07月22日

用来记录训练所用的时间
"""
import datetime


class TimeSpent:
    def __init__(self, mode):
        self.__start = datetime.datetime.now()
        self.__end = None
        self.mode = mode
        if self.mode == 'train':
            print('开始训练模型....')
        elif self.mode == 'test':
            print('开始测试模型....')
            

    # @staticmethod
    # def check_dir(dirs):
    #     if not os.path.exists(dirs):
    #         os.makedirs(dirs)

    def start_time(self):
        self.__start = datetime.datetime.now()

    def end_time(self):
        self.__end = datetime.datetime.now()

    def spent_time(self, global_step):
        """
        打印出训练使用时间
        :return: 
        """
        if self.__end is None:
            self.__end = datetime.datetime.now()

        spent_time = self.__end - self.__start
        
        if self.mode == 'train':  
            str1 = "{:*^100}".format("训练用时")
            str2 = "训练开始于 {}年{}月{}日 {}:{}:{}".format(self.__start.year,
                                                    self.__start.month,
                                                    self.__start.day,
                                                    self.__start.hour,
                                                    self.__start.minute,
                                                    self.__start.second)
            str3 = "{}次iterations训练结束于 {}年{}月{}日 {}:{}:{}".format(global_step,
                                                    self.__end.year,
                                                    self.__end.month,
                                                    self.__end.day,
                                                    self.__end.hour,
                                                    self.__end.minute,
                                                    self.__end.second)
            str4 = "{}次iterations训练总共用时 {}天 {:.0f}小时 {:.0f}分钟 {:.4f}秒".format(global_step, spent_time.days, spent_time.seconds // 3600,
                                                                spent_time.seconds % 3600 // 60,
                                                                spent_time.seconds % 3600 % 60)
            str5 = '{:*^100s}'.format('训练结束')
            
        
        else:
            str1 = "{:*^100}".format("测试用时")
            str2 = "第{}张图片测试开始于 {}年{}月{}日 {}:{}:{}".format(global_step, self.__start.year,
                                                    self.__start.month,
                                                    self.__start.day,
                                                    self.__start.hour,
                                                    self.__start.minute,
                                                    self.__start.second)
            str3 = "第{}张图片测试结束于 {}年{}月{}日 {}:{}:{}".format(global_step,
                                                    self.__end.year,
                                                    self.__end.month,
                                                    self.__end.day,
                                                    self.__end.hour,
                                                    self.__end.minute,
                                                    self.__end.second)
            str4 = "第{}张图片测试用时 {}天 {:.0f}小时 {:.0f}分钟 {:.4f}秒".format(global_step, spent_time.days, spent_time.seconds // 3600,
                                                                spent_time.seconds % 3600 // 60,
                                                                spent_time.seconds % 3600 % 60)
            str5 = '{:*^100s}'.format('测试结束')
                
            
        print(str1)
        print(str2)
        print(str3)
        print(str4)
        print(str5)

    def return_spent_time(self):
        """
        返回迅来你使用的时间的字符串，使用logger记录保存
        :return: 
        """
        if self.__end is None:
            self.__end = datetime.datetime.now()

        spent_time = self.__end - self.__start

        str1 = "{:*^100}".format("训练用时")
        str2 = "训练开始于 {}年{}月{}日 {}:{}:{}".format(self.__start.year,
                                                 self.__start.month,
                                                 self.__start.day,
                                                 self.__start.hour,
                                                 self.__start.minute,
                                                 self.__start.second)
        str3 = "训练结束于 {}年{}月{}日 {}:{}:{}".format(self.__end.year,
                                                 self.__end.month,
                                                 self.__end.day,
                                                 self.__end.hour,
                                                 self.__end.minute,
                                                 self.__end.second)
        str4 = "训练总共用时 {}天 {:.0f}小时 {:.0f}分钟 {:.4f}秒".format(spent_time.days, spent_time.seconds // 3600,
                                                             spent_time.seconds % 3600 // 60,
                                                             spent_time.seconds % 3600 % 60)
        str5 = '{:*^100s}'.format('训练结束')

        return ';'.join([str2, str3, str4])


if __name__ == '__main__':
	# 使用实例
    time0 = TimeSpent()
    time0.start_time()
    time0.end_time()
    time0.spent_time()
    print(time0.return_spent_time())  # 训练开始于 2021年8月29日 19:20:12;训练结束于 2021年8月29日 19:20:12;训练总共用时 0天 0小时 0分钟 0.0000秒

