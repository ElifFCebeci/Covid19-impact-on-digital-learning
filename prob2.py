# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from prophet import Prophet

# 1. Verileri Yükle
df_covid = pd.read_csv("covid_search_world.csv")
df_education = pd.read_csv("education_search_world.csv")

# 2. Tarih formatını düzelt
df_covid["date"] = pd.to_datetime(df_covid["date"])
df_education["date"] = pd.to_datetime(df_education["date"])

# 2.5. '<1' gibi verileri temizle
def clean_hits(x):
    if isinstance(x, str):
        if '<' in x:
            return 0.5  # '<1' gibi değerleri 0.5 kabul ediyoruz
        else:
            return float(x)  # normal stringleri sayıya çevir
    return x

df_covid["hits"] = df_covid["hits"].apply(clean_hits)
df_education["hits"] = df_education["hits"].apply(clean_hits)

# 3. Verileri birleştir (date üzerinden)
df_merged = pd.merge(df_covid, df_education, on="date", suffixes=("_covid", "_education"))

# 4. Pearson Korelasyon
correlation = df_merged["hits_covid"].corr(df_merged["hits_education"])
print(f"Pearson Korelasyon Katsayısı (Covid vs. Eğitim Aramaları): {correlation:.4f}")

# 5. T Testi: Pandemi Öncesi ve Sonrası Eğitim Arama Hacmi
pre_covid = df_merged[df_merged["date"] < "2020-03-01"]["hits_education"]
post_covid = df_merged[df_merged["date"] >= "2020-03-01"]["hits_education"]
t_stat, p_value = ttest_ind(pre_covid, post_covid, equal_var=False)

print(f"\nT Testi Sonuçları:")
print(f"T İstatistiği: {t_stat:.4f}")
print(f"P-değeri: {p_value:.4f}")

if p_value < 0.05:
    print("Sonuç: Pandemi öncesi ve sonrası arasında İSTATİSTİKSEL olarak ANLAMLI bir fark var.")
else:
    print("Sonuç: Pandemi öncesi ve sonrası arasında İSTATİSTİKSEL olarak ANLAMLI bir fark YOK.")

# 6. İteratif Çiftler Karşılaştırması
df_comparison = pd.DataFrame({
    "date": df_merged["date"],
    "Covid Hits": df_merged["hits_covid"],
    "Education Hits": df_merged["hits_education"],
    "Difference": abs(df_merged["hits_covid"] - df_merged["hits_education"])
})
print("\nİteratif Çiftler Karşılaştırması (İlk 5 Kayıt):")
print(df_comparison.head())

# 7. Zaman Serisi Analizi - Prophet Modeli (Education Aramaları için)
df_prophet = df_merged[["date", "hits_education"]].rename(columns={"date": "ds", "hits_education": "y"})
model = Prophet()
model.fit(df_prophet)

# 8. Gelecek Tahminleri
future = model.make_future_dataframe(periods=180, freq='W')
forecast = model.predict(future)

# 9. Tahminleri Görselleştir
#plt.figure(figsize=(12, 6))
model.plot(forecast)
plt.title("Dijital Eğitim Arama Hacmi - Prophet Regresyon Tahmini", fontsize=12)
plt.xlabel("Tarih")
plt.ylabel("Arama Hacmi (Hits)")
plt.axvline(pd.to_datetime("2020-03-01"), color='red', linestyle='--', label="Pandemi Başlangıcı (Mart 2020)")
plt.legend()
plt.show()