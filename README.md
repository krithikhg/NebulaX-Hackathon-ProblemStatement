# NebulaX Hackathon Repository

This repository contains the datasets, problem statements, reference materials, and submission examples for the 2026 NebulaX Hackathon organised around Land Transport Authority (LTA) themes.

## Overview

This team repository works on **Problem Statement 3: Train Condition Monitoring**. The PS1 and PS2 tracks have been removed.

## Repository Structure

```text
NebulaX-Hackathon-ProblemStatement/
├── README.md                           # Project overview and navigation guide
├── LTA_DataMall_API_User_Guide.pdf     # LTA DataMall API documentation
└── PS3/                                # Problem Statement 3
    ├── 01_Problem_Statement_3_Specifications.md
    ├── 02_Datasets/                   # Train/test datasets for each subsystem
    ├── 03_References/                 # Info kits and data documentation
    ├── 04_Example_Submission/         # Example prediction files
    ├── Door subproblem/               # Door-related exploratory work
    └── trainwhisper/                  # The app (web UI + models) and Python pipeline
```

## Problem Statement

### PS3: Train Condition Monitoring

Focus: detecting faults and estimating degradation across multiple train subsystems using sensor data and time-series models.

Start here:
- PS3/01_Problem_Statement_3_Specifications.md
- PS3/02_Datasets/
- PS3/03_References/
- PS3/04_Example_Submission/
- PS3/trainwhisper/ (app: see its README)

## LTA DataMall API

This repository also points to live transport data sources via the LTA DataMall API.

### API base URL

```text
https://datamall2.mytransport.sg/ltaodataservice/
```

### Example request

```bash
curl -X GET "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival?BusStopCode=83139" \
  -H "AccountKey: YOUR_API_KEY_HERE"
```

### Useful references
- LTA DataMall Portal: https://datamall.lta.gov.sg/content/datamall/en.html
- LTA DataMall API User Guide: ./LTA_DataMall_API_User_Guide.pdf

## Quick Start Tips

1. Read the brief for your assigned problem statement before diving into the data.
2. Inspect the example submissions to understand output formatting and expected schema.
3. Use the reference folders to understand the task domain and scoring context.
4. Validate file formats early to avoid submission mismatch.
5. Treat the README and the problem statement docs as the source of truth for the challenge.

## Resources

- [Data.gov.sg](https://data.gov.sg/)
- [LTA DataMall](https://datamall.lta.gov.sg/content/datamall/en.html)
- [SGMRT Telegram Channel](https://t.me/s/sgmrt)
- [geojson.io](http://geojson.io/)
- [Postman](https://www.postman.com/)

## Notes

This project is intended as a structured starting point for participating teams. Each problem statement folder contains the materials needed to understand the challenge and begin building a solution.

For questions about:

- **Datasets**: Review the data files and API documentation
- **API Access**: Visit [LTA DataMall Support](https://datamall.lta.gov.sg/content/datamall/en/contact-us.html)
- **Problem Statements**: Consult with hackathon organizers

!!! Mentors will be around on to help

!!! Email LTA_XX_ title your queries with [PS#] Your Question

Standard Template for PS folders:

- PSX_README.md
- data folder
- reference folder (where you store things that are for their reference/reading)
- submission folder (if needed)

---

## 📝 License

Please refer to LTA DataMall's terms of use for API data usage guidelines.

---

## 🎉 Good Luck!

We're excited to see what innovative solutions you'll build with these datasets. Happy hacking! 🚀

---
