import pandas as pd
import requests
import os

data = pd.DataFrame(pd.read_excel(r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\B题附件\附件二.xlsx'))

# print(data)

img_ID = data.ix[:,['图片链接','货号','上新日期']].values
BaseDir = r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\商品图片\按上新日期'

for img_url,filename,update in img_ID:
    update = str(update).split(' ')[0]

    # exit()
    if img_url.endswith('png'):
        print(img_url, filename, update)
        while True:
            img = requests.get(img_url)
            if img.status_code == 200:
                imgContent = img.content

                wdir = os.path.join(BaseDir,update)
                print(wdir)
                # exit()
                if not os.path.exists(wdir):
                    os.makedirs(wdir)

                with open(os.path.join(wdir,str(filename)+'.png'), 'wb') as fp:
                    fp.write(imgContent)
                break
    # if img_url.startswith('https://item'):
    #     print(filename,update)