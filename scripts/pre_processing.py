from scripts.helper_functions import *
import pandas as pd
import numpy as np
import gc

def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('data/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('data/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)

    # DATA PREPROCESSING
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)  # NaN değerleri 365243 olarak girilmiş, onlar düzeltildi
    df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)  # Araba yaş değeri boş olan gözlemler 0 olarak atandı

    # işe başlama / kişi yaşı
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # gelir kredi oranı
    df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    # Gelir miktarı/Ailedeki kişi sayıs
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    # kredi yıllık ödeme mik/gelir
    df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # kredi yıllık ödeme mik/kredi miktarı
    df['NEW_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    df['NEW_CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # Toplam kredi mik/yıllık ödeme
    df['NEW_CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    # gelir mik- yıllık ödeme
    df['NEW_INCOME_ANNUITY_DIFF'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']

    df['NEW_CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # Ürün miktarının kredi miktarının farkının gelire oranı
    df['NEW_AMT_G-C_AVERAGE_INCOME'] = (df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]) / df["AMT_INCOME_TOTAL"]

    # Kredi miktarının ürün miktarına göre farkıı
    df['NEW_CREDIT_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    # Ürün miktarının kredi miktarına göre farkı
    df['NEW_GOODS_CREDIT_DIFF'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    # Ürün miktarının gelir miktarına oranı
    df['NEW_GOODS_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']

    # kişinin yaşından çalıştığı gün farkı
    df['NEW_AGE_EMPLOYED_DIFF'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
    # Kişinin çalıştığı günün yaşına oranı
    df['NEW_EMPLOYED_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Kişinin araba yaşının çalıştığı yıla oranı
    df['NEW_CAR_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    # Kişinin araba yaşının  yaşına oranı
    df['NEW_CAR_AGE_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    # Tüm Flag değişkenlerinin toplamı
    df['NEW_FLAG_CONTACTS_SUM'] = df['FLAG_MOBIL'] + df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE'] + df[
        'FLAG_CONT_MOBILE'] + df['FLAG_PHONE'] + df['FLAG_EMAIL']  # sağladı mı sağlamadı mı 1 ve 0

    # Gelir miktarının aile üyelerine oranı (kişinin kendisi dahil)
    df['NEW_PER_INCOME'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    # Bölgelerin önem seviyelerinin toplamının gelire oranı
    df['NEW_REGIONS_RATING_INCOME_MUL'] = (df['REGION_RATING_CLIENT'] + df['REGION_RATING_CLIENT_W_CITY']) * df[
        'AMT_INCOME_TOTAL']  # gelir miktarını  2'ye bölmüştük ancak kaldırdık
    # Bölgelerin önem düzeylerinin ortalaması
    df['NEW_REGION_RATING_MEAN'] = (df['REGION_RATING_CLIENT'] + df['REGION_RATING_CLIENT_W_CITY']) / 2
    # Bölgelerin önem seviyelerinin  çarpımı
    df['NEW_REGION_RATING_AV'] = df['REGION_RATING_CLIENT'] * df['REGION_RATING_CLIENT_W_CITY']
    # Mesleklere göre maaşların ortalaması
    # print(df['ORGANIZATION_TYPE'].head())

    df['NEW_EXT_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3  # 0.765816 (tek bu olunca)

    # Bina hakkındaki bilgilerin toplamı
    df['NEW_APARTMENTS_SUM_AVG'] = df['APARTMENTS_AVG'] + df['BASEMENTAREA_AVG'] + df['YEARS_BEGINEXPLUATATION_AVG'] + \
                                   df['YEARS_BUILD_AVG'] + df['COMMONAREA_AVG'] + df['ELEVATORS_AVG'] + \
                                   df['ENTRANCES_AVG'] + df['FLOORSMAX_AVG'] + df['FLOORSMIN_AVG'] + df['LANDAREA_AVG'] + \
                                   df['LIVINGAPARTMENTS_AVG'] + df['LIVINGAREA_AVG'] + df['NONLIVINGAPARTMENTS_AVG'] + df['NONLIVINGAREA_AVG']

    df['NEW_APARTMENTS_SUM_MODE'] =  df['APARTMENTS_MODE'] + df['BASEMENTAREA_MODE'] + df['YEARS_BEGINEXPLUATATION_MODE'] + df['YEARS_BUILD_MODE'] + df['COMMONAREA_MODE'] + \
                                     df['ELEVATORS_MODE'] +df['ENTRANCES_MODE'] + df['FLOORSMAX_MODE'] + df['FLOORSMIN_MODE'] +df['LANDAREA_MODE'] + df['LIVINGAPARTMENTS_MODE'] + \
                                     df['LIVINGAREA_MODE'] +df['NONLIVINGAPARTMENTS_MODE'] + df['NONLIVINGAREA_MODE'] + df['TOTALAREA_MODE']

    # Dökümanların toplamı
    df['NEW_SUM_FLAGS_DOCUMENTS'] = df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_5'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_7'] +\
                                    df['FLAG_DOCUMENT_8'] + df['FLAG_DOCUMENT_9'] + df['FLAG_DOCUMENT_11'] + df['FLAG_DOCUMENT_13'] + \
                                    df['FLAG_DOCUMENT_14'] + df['FLAG_DOCUMENT_15'] + df['FLAG_DOCUMENT_16'] + df['FLAG_DOCUMENT_17'] + \
                                    df['FLAG_DOCUMENT_18'] + df['FLAG_DOCUMENT_19'] + df['FLAG_DOCUMENT_21']
    # Detayları bilgilerin çarpımı(değişim)
    df['NEW_DAYS_DETAILS_CHANGE_MUL'] = df['DAYS_LAST_PHONE_CHANGE'] * df['DAYS_REGISTRATION'] * df['DAYS_ID_PUBLISH']

    df["NEW_AGE"] = round(-1 * (df["DAYS_BIRTH"] / 365), 0)
    df["NEW_AGE"] = df["NEW_AGE"].astype("int")

    df['NEW_INCOME_MONTH'] = df['AMT_INCOME_TOTAL'] / 12
    df['NEW_AMT_ANNUITY_MONTH'] = df['AMT_ANNUITY'] / 12
    df['NEW_MONEY_MONTH'] = df['NEW_INCOME_MONTH'] - df['NEW_AMT_ANNUITY_MONTH']

    # FEATURE 2 - KREDİ MİKTARI / AİLEDEKİ KİŞİ SAYISI ***
    df['NEW_AMT/FAM'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']

    # FEATURE 4 - GELİR / YILLIK KREDİ * FAMILYSIZE #MODEL SONUCUNA GÖRE DEĞERLENDİR
    df["NEW_FAMILY_EFFECT"] = df['NEW_AMT/FAM'] / df['CNT_FAM_MEMBERS']

    # FEATURE 7 - KREDİ MİKTARI  / YAŞ ***
    df["NEW_CREDIT/NEW_AGE"] = df['AMT_CREDIT'] / df["NEW_AGE"]

    # FEATURE 8 - ÜRÜN / KREDİ MİKTARI ***
    df["NEW_GOODS/CREDIT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]

    # FEATURE 9 - AGE / OWN_CAR_AGE
    df["NEW_AGE/CAR_AGE"] = df["NEW_AGE"] / df["OWN_CAR_AGE"]

    # FEATURE 11 - EXT MEAN ***
    df["NEW_EXT_MEAN"] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

    # FEATURE 12 - MEW_EXT STD *** (Kaggle)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())

    # FEATURE 14 - DOKUMANLARIN TOPLAMI / DOCS ATILDI
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['NEW_DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df.drop(docs, axis=1, inplace=True)

    # FEATURE 16 NEW_PHONE_TO_BIRTH_RATIO
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    # FEATURE 17 NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']

    # FEATURE 18 - NEW_INC_ORG : Sektöründeki maaş ortalamaları
    INC_ORG = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    df['NEW_INC_ORG'] = df['ORGANIZATION_TYPE'].map(INC_ORG)

    # FEATURE 19 - NEW_PHO/ANNU : TELEFON / YILLIK ÖDEME
    df['NEW_PHO/ANNU'] = df['DAYS_LAST_PHONE_CHANGE'] / df['AMT_ANNUITY']

    # FEATURE 20 - NEW_PHO/ANNU : TELEFON / KAYIT DEĞİŞTİRME
    df["NEW_PHO/REG"] = df['DAYS_LAST_PHONE_CHANGE'] * df["DAYS_REGISTRATION"]

    # FEATURE 21 - NEW_PHO/ANNU : ADRES UYUMSUZLUKLARI
    df["NEW_FRAUD_1"] = df["REG_CITY_NOT_LIVE_CITY"] + df["REG_CITY_NOT_WORK_CITY"] + df["LIVE_CITY_NOT_WORK_CITY"]
    df["NEW_FRAUD"] = (df["NEW_FRAUD_1"] + 1) * df["DAYS_ID_PUBLISH"]
    del df["NEW_FRAUD_1"]
    df["NEW_FRAUD_std"] = (df[["REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]]).std(
        axis=1)

    # CLEAN CLASSES & LABEL ENCODING PART

    df.loc[(df["OCCUPATION_TYPE"] == "Drivers"), "OCCUPATION_TYPE"] = 1
    df.loc[(df["OCCUPATION_TYPE"] == "Waiters/barmen staff"), "OCCUPATION_TYPE"] = 1
    df.loc[(df["OCCUPATION_TYPE"] == "Low-skill Laborers"), "OCCUPATION_TYPE"] = 1
    df.loc[(df["OCCUPATION_TYPE"] == "Cleaning staff"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Sales staff"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Laborers"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Security staff"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Cooking staff"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Medicine staff"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Private service staff"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Realty agents"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Secretaries"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Accountants"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Core staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "HR staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "High skill tech staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Managers"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Medicine staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Private service staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Realty agents"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Secretaries"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "IT staff"), "OCCUPATION_TYPE"] = 4

    drop_list = ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH',
                 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'FLOORSMAX_AVG', 'LANDAREA_AVG',
                 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
                 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
                 'NONLIVINGAREA_MODE', 'TOTALAREA_MODE','FLAG_EMP_PHONE','FLAG_MOBIL', 'FLAG_CONT_MOBILE',
                 'LIVE_REGION_NOT_WORK_REGION', 'FLAG_EMAIL', 'FLAG_PHONE',
                 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
                 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                 'COMMONAREA_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
                 'NONLIVINGAREA_MODE', 'ELEVATORS_MEDI', 'EMERGENCYSTATE_MODE',
                 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', "REG_REGION_NOT_LIVE_REGION",
                 "LIVE_REGION_NOT_WORK_REGION", "AMT_REQ_CREDIT_BUREAU_HOUR", "HOUR_APPR_PROCESS_START"
                 ]

    df.drop(drop_list, axis=1, inplace=True)

    df = rare_encoder(df, 0.05)

    cat_cols, num_cols, other_cols = cols(df, "TARGET")

    df = label_encoder(df, cat_cols)
    df, app_cols = one_hot_encoder(df, nan_as_category)
    del test_df  # RAMDE yer tutmaması için oluşan dosyaları sildik
    gc.collect()
    return df

def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('data/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('data/bureau_balance.csv', nrows=num_rows)

    # Rare Encoding
    bureau.loc[(bureau["CREDIT_ACTIVE"] == "Sold"), "CREDIT_ACTIVE"] = "Closed"
    bureau.loc[(bureau["CREDIT_ACTIVE"] == "Bad debt"), "CREDIT_ACTIVE"] = "Closed"

    bureau.loc[(bureau["CREDIT_CURRENCY"] != "currency 1"), "CREDIT_CURRENCY"] = "other_currency"

    bureau = rare_encoder(bureau, 0.01)

    # NEW FUTURES
    bureau["NEW_PAID_CREDIT"] = bureau["AMT_CREDIT_SUM"] - bureau["AMT_CREDIT_SUM_DEBT"]  # ödenmiş kredi tutarı

    bureau["NEW_PAID_CREDIT_PREC"] = (bureau["NEW_PAID_CREDIT"] / bureau["AMT_CREDIT_SUM"]) * 100  # kredi ödenme oranı.

    bureau["NEW_CREDIT_CART_FLAG"] = bureau["AMT_CREDIT_SUM_LIMIT"].apply(lambda x: 1 if x > 0 else 0)  # kredi kartı var mı?

    bureau["NEW_CREDIT_DURATION"] = (bureau["DAYS_CREDIT"] * (-1)) + bureau["DAYS_CREDIT_ENDDATE"]  # kredi süresi %23 korelasyon

    bureau["NEW_CREDIT_CLOSED_PERFORMANCE"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]  # kredinin kapanması gereken zaman - kredinin kapandığı zaman (erken mi geç mi zamanında mı kapandı bilgisini verir.]

    bureau["NEW_IS_CREDIT_ACTIVE"] = bureau["CREDIT_ACTIVE"].apply(lambda x: 1 if x == "Active" else 0)  # Aktif kredilere flag atar.

    bureau["NEW_CUSTOMER_BASED_CREDIT_COUNT"] = bureau.groupby('SK_ID_CURR').agg({"NEW_IS_CREDIT_ACTIVE": 'count'})  # Müşteri bazlı ödemesi devam eden kredi sayısı.

    bureau["NEW_LATE"] = bureau["NEW_CREDIT_CLOSED_PERFORMANCE"].apply(lambda x: 1 if x < 0 else 0)  # Ödemede Gecikme var mı?

    bureau["NEW_EARLY"] = bureau["NEW_CREDIT_CLOSED_PERFORMANCE"].apply(lambda x: 1 if x > 0 else 0)  # Erken mi ödemiş?

    # BUREAU_LOAN_COUNT | Bir kişinin aldığı farklı kredi sayıları
    grp = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'NEW_BUREAU_LOAN_COUNT'})
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')

    # BUREAU_LOAN_TYPES | Bir kişinin kaç farklı tipte krediye sahip olduğu
    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'NEW_BUREAU_LOAN_TYPES'})
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')

    # AVERAGE_LOAN_TYPE| Bir kişinin aldığı farklı kredi türlerinin oranı
    bureau['NEW_AVERAGE_LOAN_TYPE'] = bureau['NEW_BUREAU_LOAN_COUNT'] / bureau['NEW_BUREAU_LOAN_TYPES']  # bir kişinin aldığı farklı kredi tipi oranı. bir kişi hep aynı türden mi kredi almış, yoksa farklı türlerden mi?

    # CREDIT_ACTIVE_BINARY | Aktif durumda olan kredilerin % si
    # Kredi aktif durumunu yeni bir değişkende gösteriyoruz. 0 kredileri kapanmış, 1 kredileri devam ediyor.
    bureau.loc[(bureau['CREDIT_ACTIVE'] == "Closed"), 'CREDIT_ACTIVE_BINARY'] = 0
    bureau.loc[(bureau['CREDIT_ACTIVE'] != "Closed"), 'CREDIT_ACTIVE_BINARY'] = 1
    bureau['CREDIT_ACTIVE_BINARY'] = bureau['CREDIT_ACTIVE_BINARY'].astype('int32')

    # NEW_ACTIVE_LOANS_PERCENTAGE | MÜŞTERİ başına AKTİF olan ortalama kredi sayısını hesaplayın
    grp = bureau.groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ACTIVE_BINARY': 'NEW_ACTIVE_LOANS_PERCENTAGE'})  # 1e yakın olması kötü oluyor
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')  # ana tablo ile birleştirme
    del bureau['CREDIT_ACTIVE_BINARY']  # gereksiz olan sütun atıldı
    gc.collect()

    # NEW_DAYS_DIFF | Bir kişi kaç gün aralıkla yeni krediler almış
    # Bir kişinin almış olduğu farklı kredileri alma günleri sıralandı
    grp = bureau[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)  # rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})
    print("Grouping and Sorting done")
    grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT'] * -1  # günler - olarak getirildiğinden + yapıldı
    grp1['NEW_DAYS_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT1'].diff()  # aldığı farklı krediler arasında kaçar gün olduğu hesaplandı
    grp1['NEW_DAYS_DIFF'] = grp1['NEW_DAYS_DIFF'].fillna(0).astype('uint32')  # ilk değişkende nan geleceği için 0 ile doldurdum. diff fonksiyonunda 2. değerden 1. değer çıkarılıyor. bu sebeple ilk değerde nan geliyor.
    del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']  # gereksiz columnlar atıldı
    print("Difference days calculated")
    # ana tablo ile birleştirme işlemleri
    bureau = bureau.merge(grp1, on=['SK_ID_BUREAU'], how='left')
    print("Difference in Dates between Previous CB applications is CALCULATED ")

    # NEW_CREDIT_ENDDATE_PERCENTAGE | ÖDEMESİ DEVAM EDEN KREDİ SAYISI ORTALAMASI
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < 0, "CREDIT_ENDDATE_BINARY"] = 0  # ödemesi bitmiş (Closed) krediler
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] >= 0, "CREDIT_ENDDATE_BINARY"] = 1  # ödemesi devam eden (Active) krediler
    grp = bureau.groupby(by=['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ENDDATE_BINARY': 'NEW_CREDIT_ENDDATE_PERCENTAGE'})  # ödemesi devam eden kredi sayısının ortalamaları
    del bureau['CREDIT_ENDDATE_BINARY']  # gereksiz olan binary columnun düşürülmesi
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')  # ana tabloya ekleme

    # NEW_AMT_PER_PAY | ÖDENEN BORÇ %Sİ
    bureau["NEW_AMT_PER_PAY"] = 1 - ((bureau["AMT_CREDIT_SUM"] - bureau["AMT_CREDIT_SUM_DEBT"]) / bureau["AMT_CREDIT_SUM"])

    # DAYS_ENDDATE_DIFF | Ödenmemiş krediler arasındaki gün farkları
    # NOT: Groupby aggregation işleminde mean ve sum alınabilir
    bureau['CREDIT_ENDDATE_BINARY'] = bureau['DAYS_CREDIT_ENDDATE']
    # Ödemesi devam edenler(+) ve ödemesi bitmiş olanlar(0 veya -) değerler belirtiliyor
    bureau.loc[(bureau["DAYS_CREDIT_ENDDATE"] <= 0), "CREDIT_ENDDATE_BINARY"] = 0  # ödemesi bitmiş (Closed) krediler
    bureau.loc[(bureau["DAYS_CREDIT_ENDDATE"] > 0), "CREDIT_ENDDATE_BINARY"] = 1  # ödemesi devam eden (Active) krediler
    # Ödemesi devam eden krediler üzerinde işlem yapılacak
    B1 = bureau[bureau['CREDIT_ENDDATE_BINARY'] == 1]
    del bureau["CREDIT_ENDDATE_BINARY"]

    # Ödemesi tamamlanmaya krediler arasındaki gün farklarının hesaplanması
    # Create Dummy Column for CREDIT_ENDDATE
    B1['DAYS_CREDIT_ENDDATE1'] = B1['DAYS_CREDIT_ENDDATE']
    # Groupby Each Customer ID
    grp = B1[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by=['SK_ID_CURR'])
    # Kredi ödeme sürelerinin kişi özelinde küçükten büyüğe sıralanması
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending=True)).reset_index(drop=True)
    del grp
    gc.collect()
    print("Grouping and Sorting done")
    # diff fonksiyonunda ilk satırlarda gelen nan valuler 0 ile dolduruldu.
    grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
    grp1['NEW_DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
    gc.collect()
    print("Difference days calculated")

    # Ana tablo ile birleştirilmesi
    bureau = bureau.merge(grp1, on=['SK_ID_BUREAU'], how='left')
    del grp1, B1
    gc.collect()

    # NEW_OVERDUE_DEBT_RATIO - Toplam Geciken Borç / Toplam Borç
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)  # nan değerler borç yok olarak alındı
    bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)  # nan değerler gecikme yok olarak alındır
    # grp1 bir kişinin toplam borcu
    # grp2 bir kişinin toplam gecikmiş borcu
    grp1 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str,columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    # ana tabloya ekleme
    bureau = bureau.merge(grp1, on=['SK_ID_CURR'], how='left')
    bureau = bureau.merge(grp2, on=['SK_ID_CURR'], how='left')
    del grp1, grp2
    gc.collect()
    bureau['NEW_OVERDUE_DEBT_RATIO'] = bureau['TOTAL_CUSTOMER_OVERDUE'] / bureau['TOTAL_CUSTOMER_DEBT']  # kişinin toplam gecikmiş borcunun, toplam borcuna oranı
    del bureau['TOTAL_CUSTOMER_OVERDUE'], bureau['TOTAL_CUSTOMER_DEBT']  # gereksiz üretilen sütunlar kaldırıldı
    gc.collect()

    # One Hot
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'SK_ID_CURR': ['count'],
        'DAYS_ENDDATE_FACT':  ['mean'],
        'DAYS_CREDIT': ['max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['sum','mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max','mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max','mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum','min'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum','mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'NEW_PAID_CREDIT': ['sum'],
        'NEW_PAID_CREDIT_PREC': ['mean'],
        'NEW_CREDIT_CART_FLAG': ['mean'],
        'NEW_CREDIT_DURATION': ['mean'],
        'NEW_CREDIT_CLOSED_PERFORMANCE': ["count", "min", "max", "mean"],
        'NEW_IS_CREDIT_ACTIVE': ["count"],
        'NEW_CUSTOMER_BASED_CREDIT_COUNT': ["mean"],
        'NEW_EARLY': ["sum"],
        'NEW_LATE': ["sum"],
        'NEW_BUREAU_LOAN_COUNT': ["min","max","sum","mean"],
        'NEW_BUREAU_LOAN_TYPES': ["min","max","sum","mean"],
        'NEW_AVERAGE_LOAN_TYPE': ["min","max","sum","mean"],
        'NEW_ACTIVE_LOANS_PERCENTAGE': ["min","max","mean"],
        'NEW_DAYS_DIFF': ["max","mean"],
        'NEW_CREDIT_ENDDATE_PERCENTAGE': ["max","mean"],
        'NEW_AMT_PER_PAY': ["min","max","sum","mean"],
        'DAYS_ENDDATE_DIFF': ["min","max","sum","mean"],
        'NEW_OVERDUE_DEBT_RATIO': ["min","max","sum","mean"]
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # NEW_EARLY_RATIO
    bureau_agg['NEW_EARLY_RATIO'] = bureau_agg['BURO_NEW_EARLY_SUM'] / bureau_agg['BURO_NEW_CREDIT_CLOSED_PERFORMANCE_COUNT'] # Erken ödeme oranı

    # NEW_LATE_RATIO
    bureau_agg['NEW_LATE_RATIO'] = bureau_agg['BURO_NEW_LATE_SUM'] / bureau_agg['BURO_NEW_CREDIT_CLOSED_PERFORMANCE_COUNT'] # Geç ödeme oranı

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('data/previous_application.csv', nrows=num_rows)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    drop_list = ['FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NAME_TYPE_SUITE', 'NAME_SELLER_INDUSTRY',
                 'WEEKDAY_APPR_PROCESS_START', 'NFLAG_INSURED_ON_APPROVAL']
    prev.drop(drop_list, axis=1, inplace=True)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # prev['APP_CREDIT_PERC'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION']

    #Rare
    # AZ RASTLANAN DEĞİŞKENLER KENDİLERİNE YAKIN SINIFLARA AKTARILDILAR.
    prev.loc[(prev[
                  "NAME_PAYMENT_TYPE"] == "Cashless from the account of the employer"), "NAME_PAYMENT_TYPE"] = "Cash through the bank "
    # CODE_REJECT_REASON
    prev.loc[
        (prev["CODE_REJECT_REASON"] != "CLIENT") & (prev["CODE_REJECT_REASON"] != "XAP"), "CODE_REJECT_REASON"] = "HC"
    # CHANNEL_TYPE
    prev.loc[(prev["CHANNEL_TYPE"] == "Regional / Local"), "CHANNEL_TYPE"] = "Regional / Local / Stone"
    prev.loc[(prev["CHANNEL_TYPE"] == "Stone"), "CHANNEL_TYPE"] = "Regional / Local / Stone"
    prev.loc[(prev["CHANNEL_TYPE"] == "Credit and cash offices"), "CHANNEL_TYPE"] = "Credit and cash offices / Channel"
    prev.loc[
        (prev["CHANNEL_TYPE"] == "Channel of corporate sales"), "CHANNEL_TYPE"] = "Credit and cash offices / Channel"
    # NAME_YIELD_GROUP
    prev.loc[(prev["NAME_YIELD_GROUP"] == "low_action"), "NAME_YIELD_GROUP"] = "low"
    prev.loc[(prev["NAME_YIELD_GROUP"] == "low_normal"), "NAME_YIELD_GROUP"] = "low"
    prev.loc[(prev["NAME_YIELD_GROUP"] == "high"), "NAME_YIELD_GROUP"] = "mid/high"
    prev.loc[(prev["NAME_YIELD_GROUP"] == "middle"), "NAME_YIELD_GROUP"] = "mid/high"
    # NAME_PORTFOLIO
    prev.loc[(prev["NAME_PORTFOLIO"] == "Cards"), "NAME_PORTFOLIO"] = "Cards/Car/Cash"
    prev.loc[(prev["NAME_PORTFOLIO"] == "Cars"), "NAME_PORTFOLIO"] = "Cards/Car/Cash"
    prev.loc[(prev["NAME_PORTFOLIO"] == "Cash"), "NAME_PORTFOLIO"] = "Cards/Car/Cash"

    # Kabul edilme ve edilmeme durumuna göre ikiye ayrıldı ve verisetini değerlendirmek amacıyla target olarak kullanıldı.
    prev.loc[(prev["NAME_CONTRACT_STATUS"] == "Approved"), "NAME_CONTRACT_STATUS"] = 0
    prev.loc[(prev["NAME_CONTRACT_STATUS"] == "Unused offer"), "NAME_CONTRACT_STATUS"] = 0
    prev.loc[(prev["NAME_CONTRACT_STATUS"] == "Canceled"), "NAME_CONTRACT_STATUS"] = 1
    prev.loc[(prev["NAME_CONTRACT_STATUS"] == "Refused"), "NAME_CONTRACT_STATUS"] = 1
    prev["NAME_CONTRACT_STATUS"] = prev["NAME_CONTRACT_STATUS"].astype("int")

    prev = rare_encoder(prev, 0.01)


    #  NEW_APP_CREDIT_PERC | İstenen kredi ve aldığı kredi oranı
    prev['NEW_APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # NEW_CREDIBILITY | müşterilerin cevap alma hızları & ONAY durumları
    prev["DAYS_DECISION2"] = prev["DAYS_DECISION"] * -1
    prev['ANS_SPEED'] = pd.cut(x=prev['DAYS_DECISION2'], bins=[0, 100, 700, 3000], labels=["Fast", "Normal", "Late"])
    del prev["DAYS_DECISION2"]

    prev.loc[((prev["NAME_CONTRACT_STATUS"] == "Approved") & (
            prev['ANS_SPEED'] == "Fast")), "NEW_CREDIBILITY"] = 5  # hızlı ve olumlu onay alanlar
    prev.loc[((prev["NAME_CONTRACT_STATUS"] == "Approved") & (
            prev['ANS_SPEED'] == "Normal")), "NEW_CREDIBILITY"] = 4  # normal ve olumlu onay alanlar
    prev.loc[((prev["NAME_CONTRACT_STATUS"] == "Approved") & (
            prev['ANS_SPEED'] == "Late")), "NEW_CREDIBILITY"] = 3  # yavaş ve olumlu onay alanlar
    prev.loc[((prev["NAME_CONTRACT_STATUS"] == "Refused") & (
            prev['ANS_SPEED'] == "Late")), "NEW_CREDIBILITY"] = 2  # yavaş ve olumsuz onay alanlar
    prev.loc[((prev["NAME_CONTRACT_STATUS"] == "Refused") & (
            prev['ANS_SPEED'] == "Normal")), "NEW_CREDIBILITY"] = 1  # normal ve olumsuz onay alanlar
    prev.loc[((prev["NAME_CONTRACT_STATUS"] == "Refused") & (
            prev['ANS_SPEED'] == "Fast")), "NEW_CREDIBILITY"] = 0  # hızlı ve olumsuz onay alanlar

    # NEW_ANN/CDT | yıllık ödemenin kredi tutarına oranı
    prev["NEW_ANN/CDT_PERC"] = prev["AMT_ANNUITY"] / prev["AMT_CREDIT"]

    # NEW_ANN/CDT | tam tutarın vadeye bölümü
    prev["NEW_CDT/PAY"] = prev["AMT_CREDIT"] / prev["CNT_PAYMENT"]

    # NEW_PAY_ABILIT" | müşteri geçmiş aylık kredi tutarının yıllık ödmeye oranı ***
    prev["NEW_PAY_ABILITY_PERC"] = prev["NEW_CDT/PAY"] / prev["AMT_ANNUITY"]

    # NEW_ABILITY | peşinat / maaş
    prev["NEW_PAY_ANN/DOWN_PERC"] = prev["AMT_ANNUITY"] / prev["AMT_DOWN_PAYMENT"]

    # NEW_GOODS_PRICE_CREDIT_RATIO
    prev['NEW_GOODS_PRICE_CREDIT_RATIO'] = prev["AMT_GOODS_PRICE"] / prev['AMT_CREDIT']

    # NEW_CREDIT_TERM_YEAR
    prev["NEW_CREDIT_TERM_YEAR"] = prev["CNT_PAYMENT"] / 12

    # NEW_INTREST_RATIO
    prev["NEW_INTREST_RATIO"] = (((prev["AMT_ANNUITY"] * prev["NEW_CREDIT_TERM_YEAR"]) - prev["AMT_CREDIT"]) / prev["AMT_CREDIT"]) * 100  # kredinin faiz oranı

    # NEW_ANNUITY_RATIO
    prev['NEW_ANNUITY_RATIO'] = prev["AMT_ANNUITY"] * prev["AMT_CREDIT"]  # yıllık ödemenin toplam krediye oranı. gerekirse agg eklenecek -- düşürdü.

    # NEW_MONTHLY_RATIO
    prev["NEW_MONTHLY_RATIO"] = prev["NEW_ANNUITY_RATIO"] * prev["CNT_PAYMENT"]  # aylık faiz oranı

    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'NEW_GOODS_PRICE_CREDIT_RATIO': ['max', 'min'],
        'NEW_CREDIT_TERM_YEAR': ['min', 'max', 'mean'],
        'NEW_INTREST_RATIO': ['max', 'min', 'mean'],
        'NEW_ANNUITY_RATIO': ['max', 'min', 'mean'],
        'NEW_MONTHLY_RATIO': ['mean'],
        'NEW_APP_CREDIT_PERC': ['min', 'max', 'mean'],
        'NEW_CREDIBILITY': ["sum", "mean"],
        'NEW_ANN/CDT_PERC': ['min', 'max', 'mean'],
        'NEW_CDT/PAY': ['min', 'max', 'mean'],
        'NEW_PAY_ABILITY_PERC': ['min', 'max', 'mean'],
        'NEW_PAY_ANN/DOWN_PERC': ['min', 'max', 'mean']
    }

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    prev_agg.columns.tolist()
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev["NAME_CONTRACT_STATUS"] == 0]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev["NAME_CONTRACT_STATUS"] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('data/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('data/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('data/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

