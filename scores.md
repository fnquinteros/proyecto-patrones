# Combinaciones de features probadas:

| features                        | validation accuracy | test accuracy           |
| ------------------------------- | ------------------- | ----------------------- |
| LBP gray 4h 8v                  | 75% val_acc         | -                       |
| LBP gray 2h 4v                  | 50% val_acc         | -                       |
| LBP red blue green 4h 8v        | 75% val_acc         | -                       |
| LBP gray 4h 8v + hog 4/4/16     | 75% val_acc         | -                       |
| LBP red 4h 8v + hog 4/4/16      | 75% val_acc (SVM)   | 71.88% test_acc (SVM)   |
| LBP red blue 4h 8v + hog 4/4/16 | 75% val_acc         | -                       |
| hog 4/4/16                      | 81.25% val_acc      | -                       |
| hog 4/8/16                      | 81.25% val_acc      | -                       |
| LBP red 4h 8v                   | 75% val_acc (SVM)   | 75% test_acc (logistic) |
