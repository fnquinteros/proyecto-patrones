# Combinaciones de features probadas:

| features                        | validation accuracy    |
| ------------------------------- | ---------------------- |
| LBP gray 4h 8v                  | 75%                    |
| LBP gray 2h 4v                  | 50%                    |
| LBP red blue green 4h 8v        | 75%                    |
| LBP gray 4h 8v + hog 4/4/16     | 75%                    |
| LBP red 4h 8v + hog 4/4/16      | **87.5% (SVM)**        |
| LBP red blue 4h 8v + hog 4/4/16 | 75%                    |
| hog 4/4/16                      | 81.25%                 |
| hog 4/8/16                      | 81.25%                 |
| LBP red 4h 8v                   | 75%                    |
