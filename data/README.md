# Cellmap Segmentation Challenge Data
Assuming you follow the instructions in the main README.md, this is where your training data should be stored. 

The data will be stored in the following structure:

```
.
├── <dataset name>
│   └── <dataset name>.zarr
│       └── recon-<number>
│           ├── em
│           │   └── fibsem-uint8
│           │       ├── s0 <-- Highest resolution scale level
│           │       ├── s1
│           │       └── ...
│           └── labels
│               └── groundtruth
│                   └── crop<number>
│                       ├── <label class 1>
│                       │   └── s0 <-- Highest resolution
│                       ├── <label class 2>
│                       ├── ...
│                       └── all <-- All labels combined
└── README.md