# GA

## Contents
This project aims to apply genetic algorithm (GA) to solve the flexible job-shop scheduling problem (FJSP) and replicates the algorithm based on relevant literature.

Additionally, a [graphical user interface (GUI)](https://github.com/pcchiu03/eGA/blob/main/code/eGA_GUI.py) is developed to allow users to easily input and adjust test parameters, select test datasets, and display the scheduling results along with the convergence plot of the algorithm.


## Files
File locations and functions within this document are shown in the following structure.
```
GA
 ┣ code                    
 ┃ ┣ eGA_FJSP.py
 ┃ ┣ eGA_GUI.py
 ┃ ┣ eGA_UML_diagram.ipynb
 ┃ ┣ settings.py
 ┃ ┗ txt_to_excel_format.py
 ┣ dataset
 ┃ ┣ Benchmark
 ┃ ┃ ┣ FJSP
 ┃ ┃ ┃ ┗ Brandimarte/
 ┃ ┃ ┗ DataSetExplanation.txt
 ┃ ┗ Sample
 ┃   ┗ eGA_FJSP_table1.xlsx
 ┣ fig                             <- the result show by figures
 ┃ ┣ convergence_plot/
 ┃ ┣ gantt_chart/
 ┃ ┗ eGA_UML_diagram.svg
 ┗ README.md
```


## Reference
Zhang, G., Gao, L., & Shi, Y. (2011). An effective genetic algorithm for the flexible job-shop scheduling problem. 
Expert Systems with Applications, 38(4), 3563–3573.
