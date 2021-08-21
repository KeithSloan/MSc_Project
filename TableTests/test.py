import numpy as np
from astropy.table import Table, join
obs = Table.read("""name    obs_date    mag_b  mag_v
                    M31     2012-01-02  17.0   17.5
                    M31     2012-01-02  17.1   17.4
                    M101    2012-01-02  15.1   13.5
                    M82     2012-02-14  16.2   14.5
                    M31     2012-02-14  16.9   17.3
                    M82     2012-02-14  15.2   15.5
                    M101    2012-02-14  15.0   13.6
                    M82     2012-03-26  15.7   16.5
                    M101    2012-03-26  15.1   13.5
                    M101    2012-03-26  14.8   14.3
                    """, format='ascii')
print(len(obs))
print(obs)
attr = Table.read("""name  uminusr logmstar metal
                     M31   0.7     6        .2
                     M101  0.6     5        .3
                     M82   0.5     4        .4
                     """, format='ascii')
print(len(attr))
print(attr)

newTab = join(obs,attr,keys='name',join_type='inner')
print(len(newTab))
print(newTab)


newTab_by_name = newTab.group_by('name')
print(newTab_by_name)

print(newTab_by_name['name','uminusr','logmstar','metal'].groups.aggregate(np.add))

