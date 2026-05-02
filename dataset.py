import pandas as pd
import numpy as np

np.random.seed(42)  # for same data every time

def create_dataset(rows=20000):
    diseases = ['Typhoid', 'Pneumonia', 'Dengue', 'Flu', 'Common Cold', 'Malaria', 'Only Fever']
    
    # Create data
    disease_list = np.random.choice(diseases, rows)
    
    fever = np.zeros(rows)
    cough = np.zeros(rows)
    sneeze = np.zeros(rows)
    weakness = np.zeros(rows)
    migraine = np.zeros(rows)
    
    # Different symptoms per disease (tighter noise so classes stay separable, RF can reach 90%+ acc)
    s_f, s_c, s_z, s_w, s_m = 0.28, 0.32, 0.36, 0.28, 0.48
    for i, d in enumerate(disease_list):
        if d == 'Typhoid':
            fever[i] = np.clip(np.random.normal(4.5, s_f), 0, 5)
            cough[i] = np.clip(np.random.normal(1.2, s_c), 0, 5)
            sneeze[i] = np.clip(np.random.normal(0.5, s_z), 0, 5)
            weakness[i] = np.clip(np.random.normal(4.2, s_w), 0, 5)
            migraine[i] = np.clip(np.random.normal(6.5, s_m), 0, 10)
        elif d == 'Pneumonia':
            fever[i] = np.clip(np.random.normal(3.8, s_f), 0, 5)
            cough[i] = np.clip(np.random.normal(4.7, s_c), 0, 5)
            sneeze[i] = np.clip(np.random.normal(2.1, s_z), 0, 5)
            weakness[i] = np.clip(np.random.normal(3.5, s_w), 0, 5)
            migraine[i] = np.clip(np.random.normal(3.0, s_m), 0, 10)
        elif d == 'Dengue':
            fever[i] = np.clip(np.random.normal(4.7, s_f), 0, 5)
            cough[i] = np.clip(np.random.normal(0.8, s_c), 0, 5)
            sneeze[i] = np.clip(np.random.normal(0.3, s_z), 0, 5)
            weakness[i] = np.clip(np.random.normal(4.5, s_w), 0, 5)
            migraine[i] = np.clip(np.random.normal(8.2, s_m), 0, 10)
        elif d == 'Flu':
            fever[i] = np.clip(np.random.normal(3.5, s_f), 0, 5)
            cough[i] = np.clip(np.random.normal(3.8, s_c), 0, 5)
            sneeze[i] = np.clip(np.random.normal(3.5, s_z), 0, 5)
            weakness[i] = np.clip(np.random.normal(3.2, s_w), 0, 5)
            migraine[i] = np.clip(np.random.normal(4.0, s_m), 0, 10)
        elif d == 'Common Cold':
            fever[i] = np.clip(np.random.normal(1.5, s_f), 0, 5)
            cough[i] = np.clip(np.random.normal(2.5, s_c), 0, 5)
            sneeze[i] = np.clip(np.random.normal(4.2, s_z), 0, 5)
            weakness[i] = np.clip(np.random.normal(1.8, s_w), 0, 5)
            migraine[i] = np.clip(np.random.normal(2.0, s_m), 0, 10)
        elif d == 'Malaria':
            fever[i] = np.clip(np.random.normal(4.6, s_f), 0, 5)
            cough[i] = np.clip(np.random.normal(1.0, s_c), 0, 5)
            sneeze[i] = np.clip(np.random.normal(0.4, s_z), 0, 5)
            weakness[i] = np.clip(np.random.normal(4.0, s_w), 0, 5)
            migraine[i] = np.clip(np.random.normal(5.5, s_m), 0, 10)
        else:  # Only Fever
            fever[i] = np.clip(np.random.normal(3.0, s_f), 0, 5)
            cough[i] = np.clip(np.random.normal(0.5, s_c), 0, 5)
            sneeze[i] = np.clip(np.random.normal(0.2, s_z), 0, 5)
            weakness[i] = np.clip(np.random.normal(1.5, s_w), 0, 5)
            migraine[i] = np.clip(np.random.normal(1.0, s_m), 0, 10)
    
    # Gender and Age (all groups included)
    gender = np.random.choice(['Male', 'Female'], rows, p=[0.52, 0.48])
    age = np.random.normal(35, 20, rows).astype(int)
    age = np.clip(age, 1, 95)  # includes kids and 60+
    
    # Learnable gender pattern (small sensor shifts) so classifier can reach high accuracy
    is_male = gender == 'Male'
    cough = np.clip(cough + np.where(is_male, 0.75, -0.75), 0, 5)
    sneeze = np.clip(sneeze + np.where(is_male, -0.55, 0.55), 0, 5)
    
    # Age signal on sensors (balanced so disease stays 90%+ and age R2 stays high)
    a = age.astype(float)
    fever = np.clip(fever + a * 0.0135, 0, 5)
    cough = np.clip(cough + a * 0.010, 0, 5)
    sneeze = np.clip(sneeze + a * 0.008, 0, 5)
    weakness = np.clip(weakness + a * 0.016, 0, 5)
    migraine = np.clip(migraine + a * 0.036, 0, 10)
    
    df = pd.DataFrame({
        'patient_id': range(1, rows+1),
        'gender': gender,
        'age': age,
        'sensor1_fever': fever,           # Sensor 1 (forehead)
        'sensor2_cough': cough,           # Sensor 2 (chest)
        'sensor3_sneeze': sneeze,         # Sensor 3 (nose)
        'sensor4_weakness': weakness,     # Sensor 4 (body)
        'migraine_pain': migraine,
        'disease': disease_list
    })
    
    df.to_csv('health_dataset_20000.csv', index=False)
    print("Dataset created successfully! File: health_dataset_20000.csv")
    print(df.head())
    print("\nDisease counts:\n", df['disease'].value_counts())
    
    return df

# Run it
if __name__ == "__main__":
    create_dataset(20000)
