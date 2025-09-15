# CGNS File Reading Functions

This document provides comprehensive documentation for CGNS (CFD General Notation System) file reading functions. These functions enable you to open, read, and extract data from CGNS files for computational fluid dynamics applications.

## Table of Contents

- [File Operations](#file-operations)
- [Zone Information Functions](#zone-information-functions)
- [Coordinate Reading Functions](#coordinate-reading-functions)
- [Flow Solution Reading Functions](#flow-solution-reading-functions)

---

## File Operations

### `open_file_read(CGNS_filename, ifile, nbases)`

**Purpose:**
Opens a CGNS file and returns the CGNS file index number and the number of bases in the file.

**Parameters:**
- `CGNS_filename` *(string)*: CGNS file name with its complete path

**Returns:**
- `ifile` *(int)*: CGNS file index number
- `nbases` *(int)*: Number of bases present in the CGNS file

**Usage Example:**
```python
ifile, nbases = CGNS.open_file_read(file_path_name)
```

---

### `close_file(ifile)`

**Purpose:**
Closes the CGNS file with the specified index.

**Parameters:**
- `ifile` *(int)*: CGNS file index number

**Returns:**
None

**Usage Example:**
```python
CGSN.close_file(ifile)
```

---

### `descriptors_read(ifile, time)`

**Purpose:**
Reads the time description associated with the CGNS file.

**Parameters:**
- `ifile` *(int)*: CGNS file index number

**Returns:**
- `time` *(float)*: Time value associated with the CGNS file

**Usage Example:**
```python
time = CGNS.descriptors_read(ifile)
```

---

## Zone Information Functions

### `nzones_read(ifile, ibase, numzones)`

**Purpose:**
Gets the number of zones in the specified CGNS file base.

**Parameters:**
- `ifile` *(int)*: CGNS file index number
- `ibase` *(int)*: Base index number, where 1 ≤ ibase ≤ nbases

**Returns:**
- `numzones` *(int)*: Number of zones present in the specified base

**Usage Example:**
```python
numzones = CGNS.nzones_read(ifile, ibase)
```

---

### `zonedim_read(ifile, ibase, izone, idim)`

**Purpose:**
Returns the index dimension for the specified CGNS zone.

**Parameters:**
- `ifile` *(int)*: CGNS file index number
- `ibase` *(int)*: Base index number, where 1 ≤ ibase ≤ nbases
- `izone` *(int)*: Zone index number, where 1 ≤ izone ≤ nzones

**Returns:**
- `idim` *(int)*: Index dimension for the zone
  - For **Structured zones**: Base cell dimension
  - For **Unstructured zones**: Always 1

**Usage Example:**
```python
idim = CGNS.zonedim_read(ifile, ibase, izone)
```

---

### `zone_size_read(ifile, ibase, izone, idim, isize, nx, ny, nz)`

**Purpose:**
Reads the zone size and returns the zone dimensions and number of points in each direction.

**Parameters:**
- `ifile` *(int)*: CGNS file index number
- `ibase` *(int)*: Base index number, where 1 ≤ ibase ≤ nbases
- `izone` *(int)*: Zone index number, where 1 ≤ izone ≤ nzones
- `idim` *(int)*: Index dimension for the zone

**Returns:**
- `isize` *(array)*: Number of vertices, cells, and boundary vertices in each index dimension
- `nx` *(int)*: Number of points in x-direction
- `ny` *(int)*: Number of points in y-direction
- `nz` *(int)*: Number of points in z-direction (for 2D meshes, nz = 0)

**Usage Example:**
```python
isize, nx, ny, nz = CGNS.zone_size_read(ifile, ibase, izone, idim)
```

---

## Coordinate Reading Functions

### `read_2D_coord(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny, x)`

**Purpose:**
Reads 2D coordinates (grid) with dimensions (nx, ny).

**Parameters:**
- `var` *(string)*: Name of the coordinate array
- `ifile` *(int)*: CGNS file index number
- `ibase` *(int)*: Base index number, where 1 ≤ ibase ≤ nbases
- `izone` *(int)*: Zone index number, where 1 ≤ izone ≤ nzones
- `ijk_min` *(array)*: Lower range index in file (e.g., imin, jmin, kmin)
- `ijk_max` *(array)*: Upper range index in file (e.g., imax, jmax, kmax)
- `nx` *(int)*: Size of coordinate array in x-direction
- `ny` *(int)*: Size of coordinate array in y-direction

**Returns:**
- `x` *(array)*: Array of coordinate (grid) values

**Usage Example:**
```python
x = CGNS.read_2D_coord(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny)
```

---

### `read_3D_coord(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny, nz, x)`

**Purpose:**
Reads 3D coordinates (grid) with dimensions (nx, ny, nz).

**Parameters:**
- `var` *(string)*: Name of the coordinate array
- `ifile` *(int)*: CGNS file index number
- `ibase` *(int)*: Base index number, where 1 ≤ ibase ≤ nbases
- `izone` *(int)*: Zone index number, where 1 ≤ izone ≤ nzones
- `ijk_min` *(array)*: Lower range index in file (e.g., imin, jmin, kmin)
- `ijk_max` *(array)*: Upper range index in file (e.g., imax, jmax, kmax)
- `nx` *(int)*: Size of coordinate array in x-direction
- `ny` *(int)*: Size of coordinate array in y-direction
- `nz` *(int)*: Size of coordinate array in z-direction

**Returns:**
- `x` *(array)*: Array of coordinate (grid) values

**Usage Example:**
```python
x = CGNS.read_3D_coord(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny, nz)
```

---

## Flow Solution Reading Functions

### `read_2D_flow(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny, q)`

**Purpose:**
Reads a specified variable from a 2D flow solution file.

**Parameters:**
- `var` *(string)*: Name of the flow variable array
- `ifile` *(int)*: CGNS file index number
- `ibase` *(int)*: Base index number, where 1 ≤ ibase ≤ nbases
- `izone` *(int)*: Zone index number, where 1 ≤ izone ≤ nzones
- `ijk_min` *(array)*: Lower range index in file (e.g., imin, jmin, kmin)
- `ijk_max` *(array)*: Upper range index in file (e.g., imax, jmax, kmax)
- `nx` *(int)*: Size of coordinate array in x-direction
- `ny` *(int)*: Size of coordinate array in y-direction

**Returns:**
- `q` *(array)*: Array of flow parameter values

**Usage Example:**
```python
q = CGNS.read_2D_flow(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny)
```

---

### `read_3D_flow(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny, nz, q)`

**Purpose:**
Reads a specified variable from a 3D flow solution file.

**Parameters:**
- `var` *(string)*: Name of the flow variable array
- `ifile` *(int)*: CGNS file index number
- `ibase` *(int)*: Base index number, where 1 ≤ ibase ≤ nbases
- `izone` *(int)*: Zone index number, where 1 ≤ izone ≤ nzones
- `ijk_min` *(array)*: Lower range index in file (e.g., imin, jmin, kmin)
- `ijk_max` *(array)*: Upper range index in file (e.g., imax, jmax, kmax)
- `nx` *(int)*: Size of coordinate array in x-direction
- `ny` *(int)*: Size of coordinate array in y-direction
- `nz` *(int)*: Size of coordinate array in z-direction

**Returns:**
- `q` *(array)*: Array of flow parameter values

**Usage Example:**
```python
q = CGNS.read_3D_flow(var, ifile, ibase, izone, ijk_min, ijk_max, nx, ny, nz)
```

---

## Notes

- **Index numbering**: All indices in CGNS follow Fortran convention (1-based indexing)
- **Memory management**: Remember to close files using `close_file()` after reading to free memory
- **Error handling**: Always check return values and implement appropriate error handling in your applications
- **Performance**: For large datasets, consider reading data in chunks rather than loading entire arrays into memory


# CGNS File Writing Functions

This document provides comprehensive documentation for CGNS (CFD General Notation System) file writing functions. These functions enable you to create and write data to CGNS files for computational fluid dynamics applications.

## Table of Contents

- [File Creation Functions](#file-creation-functions)
- [Coordinate Writing Functions](#coordinate-writing-functions)
- [Solution Writing Functions](#solution-writing-functions)

---

## File Creation Functions

### `create_file_cgns(CGNS_solnname, aux_dim)`

**Purpose:**
Creates a new CGNS solution file and sets up the bases depending on the file dimension (2D or 3D).

**Parameters:**
- `CGNS_solnname` *(string)*: CGNS file name with its complete path
- `aux_dim` *(string)*: Auxiliary dimension specifier ("2D" or "3D")

**Returns:**
None

**Usage Example:**
```python
CGNS.create_file_cgns(CGNS_solnname, aux_dim)
```

**Notes:**
- The function automatically creates appropriate bases for 2D or 3D simulations
- Ensure the directory path exists before calling this function

---

## Coordinate Writing Functions

### `write_2D_coord(CGNS_filename, zone_number, nx, ny, xcoord, ycoord)`

**Purpose:**
Writes 2D coordinates to a CGNS file.

**Parameters:**
- `CGNS_filename` *(string)*: CGNS file name with its complete path
- `zone_number` *(int)*: Zone index number, where 1 ≤ zone_number ≤ nzones
- `nx` *(int)*: Size of coordinate array in x-direction
- `ny` *(int)*: Size of coordinate array in y-direction
- `xcoord` *(array)*: Array of x-coordinate values
- `ycoord` *(array)*: Array of y-coordinate values

**Returns:**
None

**Usage Example:**
```python
CGNS.write_2D_coord(CGNS_filename, zone_number, nx, ny, xcoord, ycoord)
```

**Notes:**
- Coordinate arrays must have dimensions matching nx × ny
- The CGNS file must be created before writing coordinates

---

### `write_3D_coord(CGNS_filename, zone_number, nx, ny, nz, xcoord, ycoord, zcoord)`

**Purpose:**
Writes 3D coordinates to a CGNS file.

**Parameters:**
- `CGNS_filename` *(string)*: CGNS file name with its complete path
- `zone_number` *(int)*: Zone index number, where 1 ≤ zone_number ≤ nzones
- `nx` *(int)*: Size of coordinate array in x-direction
- `ny` *(int)*: Size of coordinate array in y-direction
- `nz` *(int)*: Size of coordinate array in z-direction
- `xcoord` *(array)*: Array of x-coordinate values
- `ycoord` *(array)*: Array of y-coordinate values
- `zcoord` *(array)*: Array of z-coordinate values

**Returns:**
None

**Usage Example:**
```python
CGNS.write_3D_coord(CGNS_filename, zone_number, nx, ny, nz, xcoord, ycoord, zcoord)
```

**Notes:**
- Coordinate arrays must have dimensions matching nx × ny × nz
- The CGNS file must be created before writing coordinates

---

## Solution Writing Functions

### `write_soln_2D(CGNS_solnname, zone_number, nx, ny, solution, var_name)`

**Purpose:**
Writes a 2D solution variable to a CGNS file.

**Parameters:**
- `CGNS_solnname` *(string)*: CGNS file name with its complete path
- `zone_number` *(int)*: Zone index number, where 1 ≤ zone_number ≤ nzones
- `nx` *(int)*: Size of solution array in x-direction
- `ny` *(int)*: Size of solution array in y-direction
- `solution` *(array)*: Array of solution variable values
- `var_name` *(string)*: Name of the solution variable

**Returns:**
None

**Usage Example:**
```python
CGNS.write_soln_2D(CGNS_solnname, zone_number, nx, ny, solution, var_name)
```

**Notes:**
- Solution array must have dimensions matching nx × ny
- Variable names should follow CGNS naming conventions
- Common variable names: "Density", "MomentumX", "MomentumY", "Pressure", "VelocityX", "VelocityY", "Temperature"

---

### `write_soln_3D(CGNS_solnname, zone_number, nx, ny, nz, solution, var_name)`

**Purpose:**
Writes a 3D solution variable to a CGNS file.

**Parameters:**
- `CGNS_solnname` *(string)*: CGNS file name with its complete path
- `zone_number` *(int)*: Zone index number, where 1 ≤ zone_number ≤ nzones
- `nx` *(int)*: Size of solution array in x-direction
- `ny` *(int)*: Size of solution array in y-direction
- `nz` *(int)*: Size of solution array in z-direction
- `solution` *(array)*: Array of solution variable values
- `var_name` *(string)*: Name of the solution variable

**Returns:**
None

**Usage Example:**
```python
CGNS.write_soln_3D(CGNS_solnname, zone_number, nx, ny, nz, solution, var_name)
```

**Notes:**
- Solution array must have dimensions matching nx × ny × nz
- Variable names should follow CGNS naming conventions
- Common variable names: "Density", "MomentumX", "MomentumY", "MomentumZ", "Pressure", "VelocityX", "VelocityY", "VelocityZ", "Temperature"

---

## General Notes

### File Writing Workflow
1. **Create the CGNS file** using `create_file_cgns()`
2. **Write coordinates** using `write_2D_coord()` or `write_3D_coord()`
3. **Write solution variables** using `write_soln_2D()` or `write_soln_3D()`

### Best Practices
- **Error handling**: Always implement proper error checking when writing files
- **Memory management**: Ensure arrays are properly allocated before writing
- **File permissions**: Verify write permissions for the target directory
- **Data consistency**: Ensure coordinate and solution array dimensions match
- **Variable naming**: Use standard CGNS variable names for better compatibility

### Common Variable Names
- **Flow variables**: "Density", "Pressure", "Temperature"
- **Momentum components**: "MomentumX", "MomentumY", "MomentumZ"
- **Velocity components**: "VelocityX", "VelocityY", "VelocityZ"
- **Turbulence**: "TurbulentEnergyKinetic", "TurbulentDissipation"
- **Custom variables**: Follow descriptive naming conventions

### Index Conventions
- **Zone numbering**: Follows Fortran convention (1-based indexing)
- **Array ordering**: Ensure proper array ordering for multi-dimensional data
- **Coordinate systems**: Right-handed coordinate system is standard
