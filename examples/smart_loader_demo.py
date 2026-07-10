from zenerestimation.data import BatteryDataset

dataset = BatteryDataset.from_csv("datasets/raw/732B-5610110.csv")
#dataset = BatteryDataset.from_csv("datasets/raw/732B-5610410.csv")

print("\nSMART LOADER DEMO")
print("-" * 40)

print("Battery ID :", dataset.metadata["battery_id"])
print("Format     :", dataset.metadata["format"])
print("Rows       :", len(dataset))
print("Columns    :", list(dataset.data.columns))

print("\nFirst five rows")
print(dataset.data.head())

#print("\nSummary")
#print(dataset.summary())

summary = dataset.summary()

print("\nSummary")
print("-" * 40)

for key, value in summary.items():
    print(f"{key:20}: {value}")