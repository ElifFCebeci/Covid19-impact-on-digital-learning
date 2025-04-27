import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri kümesini oku
df = pd.read_csv("covid_impact_education_full.csv")

# 'Date' sütununu tarih formatına çevir
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Ülkelerin eğitim durumlarını sayarak gruplandır
status_counts = df.groupby(['Date', 'Status']).size().reset_index(name='Count')

# Grafik oluştur
plt.figure(figsize=(12, 8))
sns.lineplot(data=status_counts, x='Date', y='Count', hue='Status', markers=True)

# Grafiği özelleştir
plt.title("COVID-19'un Eğitim Üzerindeki Etkisi")
plt.xlabel("Tarih")
plt.ylabel("Ülke Sayısı")
plt.xticks(rotation=45)
plt.legend(title='Durum', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Grafiği göster
plt.show()