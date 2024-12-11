Available Data
=============================

Below is a list of available data for the segmentation challenge. Yu can visualize each crop in neuroglancer by clicking on the link in the "Neuroglancer URL" column.

.. raw:: html

   <!-- Load DataTables CSS -->
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">

   <!-- Load jQuery -->
   <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

   <!-- Load DataTables JS -->
   <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>

.. raw:: html

   <table id="my-table" class="display" style="width:100%">
     <thead>
       <!-- We'll dynamically fill these from CSV headers -->
     </thead>
     <tbody>
       <!-- We'll dynamically fill these from CSV data -->
     </tbody>
   </table>

.. raw:: html

   <script>
   $(document).ready(function() {
       // Fetch the CSV file
       $.get("_static/SegmentationChallengeWithNeuroglancerURLs_20241211.csv", function(csvData) {
            // Split into lines
           var lines = csvData.trim().split("\n");
           
           // First line is headers
           var headers = lines[0].split(",");
           // Remaining lines are data rows
           var data = lines.slice(1).map(function(line) {
               return line.split(",");
           });

           // Initialize DataTable with custom render for the third column (index 2)
           $('#my-table').DataTable({
               data: data,
               columns: [
                   { title: headers[0] },
                   { title: headers[1] },
                   { 
                     title: headers[2],
                     render: function(data, type, row, meta) {
                         // 'data' is the cell content for the URL column
                         return '<a href="' + data + '" target="_blank" rel="noopener noreferrer">' + data + '</a>';
                     }
                   }
               ]
           });
       });
   });
   </script>