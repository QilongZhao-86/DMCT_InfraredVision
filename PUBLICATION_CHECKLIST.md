# Publication checklist

The code and archived model files have been organized for release. Complete
these owner decisions before making the GitHub repository public:

1. Choose the project license and confirm all code copyright holders approve
   its use.
2. The venue and acceptance are now recorded in `CITATION.cff` and the README.
   After publication, add the DOI and final volume, issue, and page/article
   number details.
3. Confirm that the three checkpoint assets may be redistributed and that the
   listed metrics may be published.

The paper-aligned Adam optimizer and dataset-specific evaluation resolution are
newer than the archived checkpoints. The code has not been run end-to-end from
a clean Linux/CUDA installation after these changes. Re-train and evaluate with
the canonical data, then compare against the manuscript before describing the
paper table as independently reproduced.
