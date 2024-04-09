def check_df(dataframe, head=5):
    print(10*"#" + " Shape ".center(9) + 10*"#")
    print(dataframe.shape)
    print(10*"#" + " Types ".center(9) + 10*"#")
    print(dataframe.dtypes)
    print(10*"#" + " Head ".center(9) + 10*"#")
    print(dataframe.head(head))
    print(10*"#" + " Tail ".center(9) + 10*"#")
    print(dataframe.tail(head))
    print(10*"#" + " NA ".center(9) + 10*"#")
    print(dataframe.isnull().sum())
    print(10*"#" + " Quantiles ".center(9) + 10*"#")
    print(dataframe.describe([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]).T)
    print(10*"#" + " Unique Values ".center(9) + 10*"#")
    print(dataframe.nunique())


def outlier_thresholds(dataframe, col_name, q1, q3):
    """
    Bir dataframe için verilen ilgili kolondaki aykırı değerleri tespit edebilmek adına üst ve alt limitleri belirlemeyi
    sağlayan fonksiyondur

    Parameters
    ----------
    dataframe: "Dataframe"i ifade eder.
    col_name: Değişkeni ifade eder.
    q1: Veri setinde yer alan birinci çeyreği ifade eder.
    q3: Veri setinde yer alan üçüncü çeyreği ifade eder.

    Returns
    -------
    low_limit, ve up_limit değerlerini return eder
    Notes
    -------
    low, up = outlier_tresholds(df, col_name) şeklinde kullanılır.
    q1 ve q3 ifadeleri yoru açıktır. Aykırı değerle 0.01 ve 0.99 değerleriyle de tespit edilebilir.

    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1,q3):
    """
    Bir dataframein verilen değişkininde aykırı gözlerimerin bulunup bulunmadığını tespit etmeye yardımcı olan
    fonksiyondur.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1,q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def grab_outliers(dataframe, col_name, index=False):
    '''

    Fonksiyon, verilen dataframe için verilen değişkende yer alan aykırı değerleri getirir. Bun fonksiyon
    "outlier_thresholds" fonksiyonunu içinde barındırdığı için bu fonksiyona bağımlılığı vardır. outlier_tresholds
    fonksiyonu tanımlanmadan kullanılamaz.

    Parameters
    ----------
    dataframe: Aykırı gözlemlerinin yakalanması istenen dataframei ifade eder.
    col_name: İlgili dataframedeki yakalanması istenen dataframede yer alan değişkeni ifade eder.
    index: Yaklanan aykırı gözlemlerin indexini ifade eder

    Returns
    -------
    Şayet "index" değeri true girilmişse yaklanan aykırı gözlemlerin indexlerini return eder

    '''

    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    """
    Bu fonksiyon kullanıcıya belirlenen üst ve alt limitlere göre aykırı değerlerden ayıklanmış bir dataframe verir
    Fonksiyonun "outlier_thresholds" fonksiyonuna bağımlılığı vardır
    Parameters
    ----------
    dataframe: Verilen dataframei ifade eder
    col_name: Dataframe'e ait değişkeni ifade eder

    Returns
    -------
    Aykırı değerlerden ayıklanmış yeni bir dataframe return eder
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, variable, q1, q3):
    """
    Up limitin üzerinde yer alan değerleri up değeri ile low limitin altında yer alan değerli ise low değerliyle
    baskılar. Bu fonksiyonun da "outlier_thresholds" fonksiyonuna bağımlılığı vardır.
    Parameters
    ----------
    dataframe: Aykırı değerli baskılanmak istenen dataframei ifade eder.
    variable: Bir başka deyişle col_name'i ifade eder. Aykırı değerleri baskılanacak olan dataframe'in ilgili
    değişkenidir.

    Returns
    -------
    Herhangi bir değer return etmez
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    """

    Bir veri setindeki eksik gözlemleri tespit etmek için kullanılan fonksiyondur. Fonksiyon kullanıcıya "n_miss" ile
    eksik gözlem sayısını "ratio" ile de eksik gözlemlerin değişkende kapladığı yeri yüzdelik olarak ifade eder

    Parameters
    ----------
    dataframe: Veri setini ifade eder
    na_name: Eksik gözlem barındıran değişkenleri ifade eder

    Returns
    -------
    Eğer na_name parametleri True olarak girildiyse eksik gözlem barındıran değişkenleri liste olarak return eder

    Notes
    -------
    Fonksiyonun numpy ve pandas kütüphanelerine bağımlılığı vardır.

    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    """

    Fonksiyon, veri setinde yer alan eksik gözlem barındıran değişkenlerin eksiklik durumlarına göre hedef değişken
    karşısındaki ortalama ve adet bilgilerini getirir

    Parameters
    ----------
    dataframe: Veri setini ifade eder.
    target: Hedef değişkeni ifade eder.
    na_columns: Eksik gözlem barındıran değişkenleri ifade eder.

    Returns
    -------
    Herhangi bir değer return etmez.

    Notes
    -------
    Fonksiyonun numpy ve pandas kütüphanelerine bağımlılığı vardır.
    """
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    """
    Fonksiyon verilen veri setindeki ilgili değişkenleri label encoding sürecine tabii tutar.

    Parameters
    ----------
    dataframe: Veri setini ifade eder.
    binary_col: Encode edilecek olaran değişkenleri ifade eder

    Returns
    -------
    Encoding işlemi yapılmiş bir şekilde "dataframe"i return eder

    Notes
    -------
    Fonksiyonun "from sklearn.preprocessing import LabelEncoder" paketine bağımlılığı bulunmaktadır.

    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """

    Veri setindeki kategorik değşkenler için one hot encoding işlemini yapar

    Parameters
    ----------
    dataframe : Veri setini ifade eder
    categorical_cols : Kategorik değişkenleri ifade eder
    drop_first : Dummy değişken tuzağına düşmemek için ilk değşşkeni siler

    Returns
    -------
    One-hot encoding işlemi yapılmış bir şekilde "dataframe"i return eder

    Notes
    -------
    Fonksiyonun "pandas" kütüphanesine bağımlılığı bulunmaktadır.
    """
    import pandas as pd
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def cat_summary(dataframe, col_name, plot=False):
    """

    Fonksiyon, veri setinde yer alan kategorik, numerik vs... şeklinde gruplandırılan değişkenler için özet bir çıktı
    sunar.

    Parameters
    ----------
    dataframe : Veri setini ifade
    col_name : Değişken grubunu ifade eder
    plot : Çıktı olarak bir grafik istenip, istenmediğini ifade eder, defaul olarak "False" gelir

    Returns
    -------
    Herhangi bir değer return etmez

    Notes
    -------
    Fonksiyonun pandas, seaborn ve matplotlib kütüphanelerine bağımlılığı vardır.

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def rare_analyser(dataframe, target, cat_cols):
    """
    Verilen veri setindeki hedef değişkene göre değişken grubundaki nadir gözlemleri analiz eder
    Parameters
    ----------
    dataframe : Veri setini ifade eder.
    target : Hedef değişkeni ifade eder.
    cat_cols : Değişken grubunu ifade eder

    Returns
    -------
    Herhangi bir değer retrun etmez.
    """
    import pandas as pd
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    """

    Verilen veri setinde, önceden verilen orana göre rare encoding işlemi yapar

    Parameters
    ----------
    dataframe : Veri setini ifade eder.
    rare_perc : Nadir görülme oranını ifade eder.

    Returns
    -------
    Rare encoding yapılmış datafremi return eder
    """
    import numpy as np
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data
