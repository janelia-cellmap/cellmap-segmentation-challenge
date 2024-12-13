Available Data
=============================

The training crops used in the CellMap Segmentation Challenge are derived from 23 distinct eFIB-SEM datasets, including 9 cell culture samples and 14 tissue samples.
These samples come from a variety of organisms, including mouse (10 datasets), Drosophila (2 datasets), zebrafish (1 dataset), and human (1 dataset). 
The tissue types span key biological regions and organs such as the brain, heart, liver, kidney, and pancreas, providing a broad spectrum of biological contexts for participants to work with.

Below is a list of available data for the segmentation challenge. You can visualize crops groupped by collection in neuroglancer by clicking on the link in the "Neuroglancer URL" column.

.. raw:: html

   <!-- Load DataTables CSS -->
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">

   <!-- Load jQuery -->
   <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

   <!-- Load DataTables JS -->
   <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>

.. raw:: html

   <table id="main-table" class="display" style="width:100%">
     <thead>
       <!-- We'll dynamically fill these from CSV headers -->
     </thead>
     <tbody>
       <!-- We'll dynamically fill these from CSV data -->
     </tbody>
   </table>

Each crop can be visualized separately in this table by clicking on the "Neuroglancer URL" link in the table below.

.. raw:: html

   <details>
   <summary>Click to expand</summary>

   <table id="side-table" class="display" style="width:100%">
     <thead>
       <!-- We'll dynamically fill these from CSV headers -->
     </thead>
     <tbody>
       <!-- We'll dynamically fill these from CSV data -->
     </tbody>
   </table>
   </details>

.. raw:: html

   <script>
   $(document).ready(function() {
       // Fetch the CSV file
       $.get("_static/SegmentationChallengeWithNeuroglancerURLs_20241211_all_s3.csv", function(csvData) {
            // Split into lines
           var lines = csvData.trim().split("\n");
           
           // First line is headers
           var headers = lines[0].split(",");
           // Remaining lines are data rows
           var data = lines.slice(1).map(function(line) {
               return line.split(",");
           });

           // Initialize DataTable with custom render for the third column (index 2)
           $('#main-table').DataTable({
               data: data,
               columns: [
                   { title: headers[0] },
                   { title: headers[1] ,
                     render: function(data, type, row, meta) {
                         // 'data' is the cell content for the URL column
                         return '<a href="' + data + '" target="_blank" rel="noopener noreferrer">Neuroglancer Link</a>';
                     }
                   }
               ]
           });
       });

       $.get("_static/SegmentationChallengeWithNeuroglancerURLs_20241211_s3.csv", function(csvData) {
            // Split into lines
           var lines = csvData.trim().split("\n");
           
           // First line is headers
           var headers = lines[0].split(",");
           // Remaining lines are data rows
           var data = lines.slice(1).map(function(line) {
               return line.split(",");
           });

           // Initialize DataTable with custom render for the third column (index 2)
           $('#side-table').DataTable({
               data: data,
               columns: [
                   { title: headers[0] },
                   { title: headers[1] },
                   { 
                     title: headers[2],
                     render: function(data, type, row, meta) {
                         // 'data' is the cell content for the URL column
                         return '<a href="' + data + '" target="_blank" rel="noopener noreferrer">Neuroglancer Link</a>';
                     }
                   }
               ]
           });
       });
   });
   </script>