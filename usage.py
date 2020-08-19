# 用法示例

from reb import P

# 数字
# ic for "In Chars"
num = P.ic('1234567890')

# 非标点
# nic for "Not In Chars"
nopunc = P.nic('，。、')

# 表示医院的模式
# n01 for "repeat N times 0 or 1"
# n for "repeat N times"
hosp = P.tag(
    P.n(nopunc, 3, 12) + P.ic('院心') + P.n01('附属' + P.n(P.ANYCHAR, 3, 10) + P.ic('院心')),
    tag='医院')

# 表示日期的模式
date = P.n(num, 2, 4) + P.ic('-,.年') + P.n(num, 1, 2) + P.ic('-,.月') + P.n01(P.n(num, 1, 2) + '日')

# 一些检查事件
# any for "matches if ANY sub pattern matches"
event = P.any('胸部CT', ('PET' + P.n01('-') + 'CT'), '增强CT', '胸部平扫', '肠镜', '穿刺', 'B超')

# 总结一些模式，从中提取医院
ptn = (
    P.example(
        P.tag('当地医院', tag='医院'),
            '...1周前当地医院复查提示...'
    )

    # "|" for clause, each clause must has at least one example
    | P.example(
        ('就' + P.any('诊', '医') + P.n01('于') + hosp),
            '...无声嘶呛咳等不适，就诊于嘉兴市第一医院，2018-7-6查喉镜...',
      )

    | P.example(
        (P.ic('到至在于往') + P.n01(date) + hosp + '就' + P.ic('诊医')),
            '...2019.5出现左下肢水肿，6月于当地医院就诊，2019-07-29邵逸...',
            '...患者因“绝经后阴道流血1月余”于2019-4-16杭州市一医院就诊，行B超提示：...',
      )

    | P.example(
        (P.n01(date) + hosp + event + P.n(nopunc, 0, 4) + '示'),
            '...患者因“发现左侧腹股沟包块2月余”2014年12月3日台州医院B超示：左侧腹股沟多发肿大淋巴结...',
      )

    | P.example(
        (P.ic('到至在于往') + hosp + '体检'),
            '...患者2019.8.20在萧山医院体检发现...'
      )

    | P.example(
        ('入住' + hosp),
            '...于2019.7.8入住萧山医院，B超示..'
      )

    | P.example(
        (P.ic('到至在于往') + P.n01(date) + hosp + P.n(nopunc, 0, 3) + P.ic('行查检')),
            '...质地较硬，于2019-8-11台州市中心医院查彩超：...'
      )
)

# 试运行一例
example = '2018年6月患者因“大便次数增多伴便中带血”就诊于浙江金华中心医院，完善检查后诊断为乙状结肠癌'


pt = ptn.extract(example)

# 从 parse tree 中提取出标为 “医院” 的片段
for p in pt:
    print(list(p.fetch(tag='医院')))