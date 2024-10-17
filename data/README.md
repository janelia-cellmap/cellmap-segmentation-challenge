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
```

## Test Data
The test data will be stored alongside the training data with `test` as the only class label, which contains empty data. In case the regions of interest used for testing change, you will need to update the test data accordingly. If you are using the default data path, you can simply run the following from the root of the repository to do this:

```bash
csc fetch-data --crops test
```