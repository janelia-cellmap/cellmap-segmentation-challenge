========================================
Annotation Classes
========================================

This document provides an overview of the annotation classes used in our dataset. Each zarr crop contains multiple scales for different organelles, if annotated. Additionally, there is a special multiscale named ``all`` that aggregates all annotated classes.

Classes are categorized as either **atomic classes** or **group classes**:

- **Atomic Classes** represent individual organelle components.
- **Group Classes**  are sets of atomic classes grouped under a common category.

Group classes reference their constituent atomic classes through the ``group_id``.

Table of Contents
-----------------
- `Multiscale Structure`
- `Class Categories`
  - `Atomic Classes`
  - `Group Classes`
- `Detailed Class Descriptions`
- `Aliases`
- `Examples`

Multiscale Structure
--------------------
In each zarr crop, annotations are organized into multiple scales:
- **Organelle-specific scales**: Each annotated organelle has its own scale.
- **All-inclusive scale**: The ``all`` scale includes annotations for all classes.

This hierarchical structure facilitates efficient data navigation and retrieval based on specific or broad annotation needs.

Class Categories
----------------

**Atomic Classes**
Atomic classes represent the fundamental components of cellular structures. Each atomic class is defined with a unique identifier and descriptive properties.
**Group Classes**
Group classes define broader categories by combining multiple atomic classes into a set. They are useful for high-level analysis and visualization.

Detailed Class Descriptions
---------------------------

Below is a comprehensive list of all annotation classes, categorized into atomic and group classes.
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

.. raw:: html

   <script>
   $(document).ready(function() {
       // Fetch the CSV file
       $.get("_static/classes.csv", function(csvData) {
           // Split into lines
           var lines = csvData.trim().split("\n");
           
           // First line is headers
           var headers = lines[0].split(",");
           
           // Remaining lines are data rows
           var data = lines.slice(1).map(function(line) {
               return line.split(",");
           });

           // Initialize DataTable
           $('#main-table').DataTable({
               data: data,
               columns: [
                   { title: headers[0] },  // field_name
                   { title: headers[1] },  // class_id
                   { title: headers[2] },  // group_id
                   { title: headers[3] },  // long_name
                   { title: headers[4] }   // challenge
               ],
               rowCallback: function(row, rowData) {
                   // rowData[4] refers to the 'challenge' column (index 4)
                   if (rowData[4] === 'True') {
                       $(row).css('background-color', 'green');
                   }
               }
           });
       });
   });
   </script>


   
Aliases
-------
Some classes may have aliases for compatibility or alternative naming conventions. These are listed in the ``Alias`` column of the table above.

Examples
--------
- **Nucleus** (`nuc`): Comprised of multiple components including the nuclear envelope membrane (`ne_mem`), nuclear pores (`np_out`, `np_in`), heterochromatin (`hchrom`), euchromatin (`echrom`), nucleoplasm (`nucpl`), and nucleolus (`nucleo`).
- **Mitochondria** (`mito`): Includes the mitochondrial membrane (`mito_mem`), mitochondrial lumen (`mito_lum`), and mitochondrial ribosome (`mito_ribo`).
- **Endoplasmic Reticulum** (`er`): A collective class that encompasses various ER components such as the ER membrane (`er_mem`), ER lumen (`er_lum`), and ER exit site (`eres_mem`, `eres_lum`).

For detailed information on each class and their relationships, refer to the table above.