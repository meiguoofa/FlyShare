---
layout:     post
title:      Python学习-DAY14
subtitle:   Task14-datetime
date:       2020-08-05
author:     flyshare
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - cs231n
---

# datetime模块

datetime 是 Python 中处理日期的标准模块，它提供了 4 种对日期和时间进行处理的类：**datetime**、**date**、**time** 和 **timedelta**。


---

## 1. datetime类

```python
class datetime(date):
    def __init__(self, year, month, day, hour, minute, second, microsecond, tzinfo)
        pass
    def now(cls, tz=None):
        pass
    def timestamp(self):
        pass
    def fromtimestamp(cls, t, tz=None):
        pass
    def date(self):
        pass
    def time(self):
        pass
    def year(self):
        pass
    def month(self):
        pass
    def day(self):
        pass
    def hour(self):
        pass
    def minute(self):
        pass
    def second(self):
        pass
    def isoweekday(self):
        pass
    def strftime(self, fmt):
        pass
    def combine(cls, date, time, tzinfo=True):
        pass
```

- `datetime.now(tz=None)` 获取当前的日期时间，输出顺序为：年、月、日、时、分、秒、微秒。
- `datetime.timestamp()` 获取以 1970年1月1日为起点记录的秒数。
- `datetime.fromtimestamp(tz=None)` 使用 unixtimestamp 创建一个 datetime。


```python
import datetime

dt = datetime.datetime(year=2020, month=6, day=25, hour=11, minute=23, second=59)
print(dt)  # 2020-06-25 11:23:59
print(dt.timestamp())  # 1593055439.0

dt = datetime.datetime.fromtimestamp(1593055439.0)
print(dt)  # 2020-06-25 11:23:59
print(type(dt)) # <class 'datetime.datetime'>

dt = datetime.datetime.now()
print(dt)  # 2020-06-25 11:11:03.877853
print(type(dt))  # <class 'datetime.datetime'>
```

- `datetime.strftime(fmt)` 格式化 datetime 对象。

符号 | 说明
:---:|---
`%a` | 本地简化星期名称（如星期一，返回 Mon）
`%A` | 本地完整星期名称（如星期一，返回 Monday）
`%b` | 本地简化的月份名称（如一月，返回 Jan）
`%B` | 本地完整的月份名称（如一月，返回 January）
`%c` | 本地相应的日期表示和时间表示
`%d` | 月内中的一天（0-31）
`%H` | 24小时制小时数（0-23）
`%I` | 12小时制小时数（01-12）
`%j` | 年内的一天（001-366）
`%m` | 月份（01-12）
`%M` | 分钟数（00-59）
`%p` | 本地A.M.或P.M.的等价符
`%S` | 秒（00-59）
`%U` | 一年中的星期数（00-53）星期天为星期的开始
`%w` | 星期（0-6），星期天为星期的开始
`%W` | 一年中的星期数（00-53）星期一为星期的开始
`%x` | 本地相应的日期表示
`%X` | 本地相应的时间表示
`%y` | 两位数的年份表示（00-99）
`%Y` | 四位数的年份表示（0000-9999）
`%Z` | 当前时区的名称（如果是本地时间，返回空字符串）
`%%` | %号本身


如何将 datetime对象转换为任何格式的日期？


```python

import datetime

dt = datetime.datetime(year=2020, month=6, day=25, hour=11, minute=51, second=49)
s = dt.strftime("'%Y/%m/%d %H:%M:%S")
print(s)  # '2020/06/25 11:51:49

s = dt.strftime('%d %B, %Y, %A')
print(s)  # 25 June, 2020, Thursday

import datetime
dt = datetime.datetime.now()
s = dt.strftime("%Y/%m/%d %H:%M:%S")
print(s) #2020/08/05 12:11:01

```

- `datetime.date()` Return the date part.
- `datetime.time()` Return the time part, with tzinfo None.
- `datetime.year` 年
- `datetime.month` 月
- `datetime.day` 日
- `datetime.hour` 小时
- `datetime.minute` 分钟
- `datetime.second` 秒
- `datetime.isoweekday` 星期几

datetime 对象包含很多与日期时间相关的实用功能。

```python
import datetime

dt = datetime.datetime(year=2020, month=6, day=25, hour=11, minute=51, second=49)
print(dt.date())  # 2020-06-25
print(type(dt.date()))  # <class 'datetime.date'>
print(dt.time())  # 11:51:49
print(type(dt.time()))  # <class 'datetime.time'>
print(dt.year)  # 2020
print(dt.month)  # 6
print(dt.day)  # 25
print(dt.hour)  # 11
print(dt.minute)  # 51
print(dt.second)  # 49
print(dt.isoweekday())  # 4
```

在处理含有字符串日期的数据集或表格时，我们需要一种自动解析字符串的方法，无论它是什么格式的，都可以将其转化为 datetime 对象。这时，就要使用到 dateutil 中的 parser 模块。

- `parser.parse(timestr, parserinfo=None, **kwargs)` 

 如何在 python 中将字符串解析为 datetime对象？
 
 `from dateutil import parser``
 
 ```python
from dateutil import parser

s = '2020-06-25'
dt = parser.parse(s)
print(dt)  # 2020-06-25 00:00:00
print(type(dt))  # <class 'datetime.datetime'>

s = 'March 31, 2010, 10:51pm'
dt = parser.parse(s)
print(dt)  # 2010-03-31 22:51:00
print(type(dt))  # <class 'datetime.datetime'>
```

```python
print(parser.parse(s1))
print(parser.parse(s2))
print(parser.parse(s3))

#2010-01-01 00:00:00
#2000-01-31 00:00:00
#1996-10-10 22:40:00
```

计算以下列表中连续的天数。

```python
# 输入
['Oct, 2, 1869', 'Oct, 10, 1869', 'Oct, 15, 1869', 'Oct, 20, 1869','Oct, 23, 1869']

# 输出
[8, 5, 5, 3]
```

```python
import numpy as np
from dateutil import parser

dateString = ['Oct, 2, 1869', 'Oct, 10, 1869', 'Oct, 15, 1869', 'Oct, 20, 1869', 'Oct, 23, 1869']
dates = [parser.parse(i) for i in dateString]
td = np.diff(dates)
print(td)
# [datetime.timedelta(days=8) datetime.timedelta(days=5)
#  datetime.timedelta(days=5) datetime.timedelta(days=3)]
d = [i.days for i in td]
print(d)  # [8, 5, 5, 3]
```


---

## 2. date类
```python
class date:
    def __init__(self, year, month, day):
        pass
    def today(cls):
        pass
```

如何在 Python 中获取当前日期和时间？

```python
import datetime

d = datetime.date(2020, 6, 25)
print(d)  # 2020-06-25
print(type(d))  # <class 'datetime.date'>

d = datetime.date.today()
print(d)  # 2020-06-25
print(type(d))  # <class 'datetime.date'>
```

如何统计两个日期之间有多少个星期六？

```python
# 输入
d1 = datetime.date(1869, 1, 2)
d2 = datetime.date(1869, 10, 2)

# 输出
40
```

```python

d1 = datetime.date(1869, 1, 2)
d2 = datetime.date(1869, 10, 2)
dt = (d2-d1).days
print(dt)
print(dt.isoweekday()) # 星期几
# 查看星期几，如果是小于星期星期日的话则除以7+1
print(dt//7 + 1)

```

---
## 3. time类


```python
class time:
    def __init__(self, hour, minute, second, microsecond, tzinfo):
        pass
```

```python
import datetime

t = datetime.time(12, 9, 23, 12980)
print(t)  # 12:09:23.012980
print(type(t))  # <class 'datetime.time'>
```

【练习】如何将给定日期转换为当天开始的时间？
```python
# 输入
import datetime
date = datetime.date(2019, 10, 2)

# 输出
2019-10-02 00:00:00
```

```python

import datetime
date = datetime.date(2019, 10, 2)
dt = datetime.datetime(date.year,date.month,date.day)

dt = datetime.datetime.combine(date,datetime.time.combine)



```

---
## 4. timedelta类

`timedelta` 表示**具体时间实例中的一段时间**。你可以把它们简单想象成两个日期或时间之间的间隔。

它常常被用来从 `datetime` 对象中添加或移除一段特定的时间。

```python
class timedelta(SupportsAbs[timedelta]):
    def __init__(self, days, seconds, microseconds, milliseconds, minutes, hours, weeks,):
        pass
    def days(self):
        pass
    def total_seconds(self):
        pass
```

如何使用 datetime.timedelta() 类？

```python
import datetime

td = datetime.timedelta(days=30)
print(td)  # 30 days, 0:00:00
print(type(td))  # <class 'datetime.timedelta'>
print(datetime.date.today())  # 2020-07-01
print(datetime.date.today() + td)  # 2020-07-31

dt1 = datetime.datetime(2020, 1, 31, 10, 10, 0)
dt2 = datetime.datetime(2019, 1, 31, 10, 10, 0)
td = dt1 - dt2
print(td)  # 365 days, 0:00:00
print(type(td))  # <class 'datetime.timedelta'>

td1 = datetime.timedelta(days=30)  # 30 days
td2 = datetime.timedelta(weeks=1)  # 1 week
td = td1 - td2
print(td)  # 23 days, 0:00:00
print(type(td))  # <class 'datetime.timedelta'>
```

```python

import datetime
from dateutil import parser

dt1 = datetime.date.today()
str = "Dec 31, 1997"
dt2 = parser.parse(str).date()
dt3 = datetime.date(dt1.year,dt2.month,dt2.day)
print(dt1)
print(dt2)
dt = dt1 - dt2
print(dt.days)
dt = dt3 - dt1
print(dt.days)
print(dt.days * 24 * 60 * 60)


#2020-08-05
#1997-12-31
#8253
#148
#12787200


```

#### 作业


1、假设你获取了用户输入的日期和时间如`2020-1-21 9:01:30`，以及一个时区信息如`UTC+5:00`，均是`str`，请编写一个函数将其转换为timestamp：

题目说明:

```python
"""
   
Input file
example1: dt_str='2020-6-1 08:10:30', tz_str='UTC+7:00'
example2: dt_str='2020-5-31 16:10:30', tz_str='UTC-09:00'
   
Output file
result1: 1590973830.0
result2: 1590973830.0
"""
   
   
def to_timestamp(dt_str, tz_str):
    # your code here
        pass
```

2、编写Python程序以选择指定年份的所有星期日。

题目说明:

```python
"""
   
Input file
   2020
   
Output file
   2020-01-05                         
   2020-01-12              
   2020-01-19                
   2020-01-26               
   2020-02-02     
   -----
   2020-12-06               
   2020-12-13                
   2020-12-20                
   2020-12-27 
"""
   
def all_sundays(year):
    # your code here
    
```


```python

#example1: dt_str='2020-6-1 08:10:30', tz_str='UTC+7:00'
#example2: dt_str='2020-5-31 16:10:30', tz_str='UTC-09:00'

import re
from datetime import datetime, timezone, timedelta
from dateutil import parser


def func(str1,str2):
  dt1 = parser.parse(str1)
  utc_d = dt1.utcnow().replace(tzinfo=timezone.utc)
  zone = re.match(r'UTC([+|-][\d]{1,2}):00', str2).group(1)
  #print(zone)
  utc_d = utc_d.astimezone(timezone(timedelta(hours=int(zone))))
  print(utc_d.timestamp())


dt_str='2020-6-1 08:10:30'
tz_str='UTC+7:00'
func(dt_str,tz_str)
dt_str='2020-5-31 16:10:30'
tz_str='UTC-09:00'
func(dt_str,tz_str)

#1596636108.868876
#1596636108.869161

```


2.


```python
from datetime import datetime,timezone,timedelta
def func(year):
  dt1 = datetime(year=year,month=1,day=1)
  dt2 = datetime(year=1,month=12,day=31)
  dt = (dt2 - dt1).days
  d = timedelta(days=1)
  for i in range(dt+1):
    a = dt1.isoweekday()
    if a == 7:
        s = dt1.strftime("%Y-%m-%d")
        print(s)
    dt1 = dt1 + d

func(2020)

```