use numpy::ndarray::{Array2, Dim};
use numpy::PyArrayMethods;
use numpy::{IntoPyArray, PyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use rand::prelude::*;

fn gradient_grid_set_edges(
    mut grid: Vec<Vec<(f32, f32)>>,
    fixed_gradients_top: Option<Vec<(f32, f32)>>,
    fixed_gradients_bot: Option<Vec<(f32, f32)>>,
    fixed_gradients_right: Option<Vec<(f32, f32)>>,
    fixed_gradients_left: Option<Vec<(f32, f32)>>,
) -> Vec<Vec<(f32, f32)>> {
    let grid_size: usize = grid.len();

    if let Some(fixed_gradients_top) = fixed_gradients_top {
        for x in 0..grid_size {
            grid[0][x] = fixed_gradients_top[x];
        }
    }
    if let Some(fixed_gradients_bot) = fixed_gradients_bot {
        for x in 0..grid_size {
            grid[grid_size - 1][x] = fixed_gradients_bot[x];
        }
    }
    if let Some(fixed_gradients_right) = fixed_gradients_right {
        for y in 0..grid_size {
            grid[y][grid_size - 1] = fixed_gradients_right[y];
        }
    }
    if let Some(fixed_gradients_left) = fixed_gradients_left {
        for y in 0..grid_size {
            grid[y][0] = fixed_gradients_left[y];
        }
    }
    grid
}

fn generate_circular_gradient_grid(
    grid_size: usize,
    axis: &str,
    seed: u32,
) -> Vec<Vec<(f32, f32)>> {
    let mut grid: Vec<Vec<(f32, f32)>> = generate_random_gradient_grid(grid_size, seed);
    match axis {
        "horizontal" => {
            for y in 0..grid_size {
                grid[y][0] = grid[y][grid_size];
            }
        }
        "vertical" => {
            for x in 0..grid_size {
                grid[0][x] = grid[grid_size][x];
            }
        }
        "both" => {
            for y in 0..grid_size {
                grid[y][0] = grid[y][grid_size];
            }
            for x in 0..grid_size {
                grid[0][x] = grid[grid_size][x];
            }
        }
        _ => {}
    }
    grid
}

fn generate_random_gradient_grid(grid_size: usize, seed: u32) -> Vec<Vec<(f32, f32)>> {
    let mut rng: StdRng = StdRng::seed_from_u64(seed as u64);

    let mut grid: Vec<Vec<(f32, f32)>> = Vec::with_capacity(grid_size + 1);
    for _ in 0..=grid_size {
        let mut row: Vec<(f32, f32)> = Vec::with_capacity(grid_size + 1);
        for _ in 0..=grid_size {
            let angle: f32 = rng.gen::<f32>() * std::f32::consts::TAU;
            row.push((angle.cos(), angle.sin()));
        }
        grid.push(row);
    }

    grid
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn dot_product(gradient: (f32, f32), x: f32, y: f32) -> f32 {
    gradient.0 * x + gradient.1 * y
}

fn perlin_noise(grid: &Vec<Vec<(f32, f32)>>, x: f32, y: f32) -> f32 {
    let x0: usize = x.floor() as usize;
    let y0: usize = y.floor() as usize;
    let x1: usize = x0 + 1;
    let y1: usize = y0 + 1;

    let dx: f32 = x - x.floor();
    let dy: f32 = y - y.floor();

    let g00: (f32, f32) = grid[y0][x0];
    let g10: (f32, f32) = grid[y0][x1];
    let g01: (f32, f32) = grid[y1][x0];
    let g11: (f32, f32) = grid[y1][x1];

    let dot00: f32 = dot_product(g00, dx, dy);
    let dot10: f32 = dot_product(g10, dx - 1.0, dy);
    let dot01: f32 = dot_product(g01, dx, dy - 1.0);
    let dot11: f32 = dot_product(g11, dx - 1.0, dy - 1.0);

    let u: f32 = fade(dx);
    let v: f32 = fade(dy);

    let lerp_x0: f32 = dot00 + u * (dot10 - dot00);
    let lerp_x1: f32 = dot01 + u * (dot11 - dot01);

    lerp_x0 + v * (lerp_x1 - lerp_x0)
}

fn perlin_noise_with_octaves(
    grids: &Vec<Vec<Vec<(f32, f32)>>>,
    x: f32,
    y: f32,
    octaves: &Vec<f64>,
) -> f32 {
    let mut total: f32 = 0.0;
    let mut frequency: f32 = 1.0;
    for (grid, amplitude) in grids.iter().zip(octaves.iter()) {
        total += perlin_noise(grid, x * frequency, y * frequency) * (*amplitude as f32);
        frequency *= 2.0;
    }
    total
}

pub fn create_grids(
    octaves: usize,
    grid_size: usize,
    seed: u32,
    circular: bool,
    axis: &str,
) -> Vec<Vec<Vec<(f32, f32)>>> {
    let mut grids: Vec<Vec<Vec<(f32, f32)>>> = Vec::new();
    for idx in 0..octaves {
        let new_grid_size: usize = (grid_size as u64 * 2_u64.pow(idx as u32)) as usize;
        grids.push(get_gradient_grid(new_grid_size, seed, circular, axis));
    }
    grids
}

pub fn get_perlin_array(
    grids: &Option<Vec<Vec<Vec<(f32, f32)>>>>,
    grid_size: usize,
    width: u32,
    height: u32,
    octaves: Vec<f64>,
    circular: bool,
    axis: &str,
    seed: u32,
) -> Vec<Vec<f32>> {
    let used_grids: Vec<Vec<Vec<(f32, f32)>>> = match grids {
        Some(existing_grids) => existing_grids.clone(),
        None => {
            let grids: Vec<Vec<Vec<(f32, f32)>>> =
                create_grids(octaves.len(), grid_size, seed, circular, axis);
            grids
        }
    };
    let grid_size : usize = used_grids[0].len() - 1;
    let mut img: Vec<Vec<f32>> = Vec::with_capacity(height as usize);

    for x in 0..width {
        let mut row: Vec<f32> = Vec::with_capacity(height as usize);
        for y in 0..height {
            let px: f32 = x as f32;
            let py: f32 = y as f32;
            let noise_value: f32 = perlin_noise_with_octaves(
                &used_grids,
                px / width as f32 * grid_size as f32,
                py / height as f32 * grid_size as f32,
                &octaves,
            );
            row.push(noise_value);
        }
        img.push(row);
    }
    img
}

fn get_gradient_grid(
    grid_size: usize,
    seed: u32,
    circular: bool,
    axis: &str,
) -> Vec<Vec<(f32, f32)>> {
    if circular {
        generate_circular_gradient_grid(grid_size, axis, seed)
    } else {
        generate_random_gradient_grid(grid_size, seed)
    }
}

#[pyfunction]
#[pyo3(signature = (grid_size, width, height, octaves, circular, axis, seed))]
fn get_perlin_numpy(
    py: Python,
    grid_size: usize,
    width: u32,
    height: u32,
    octaves: Py<PyList>,
    circular: bool,
    axis: &str,
    seed: u32,
) -> PyResult<Py<PyArray2<f32>>> {
    let octaves: Vec<f64> = octaves.extract(py)?;
    let array: Vec<Vec<f32>> = get_perlin_array(
        &None, grid_size, width, height, octaves, circular, axis, seed,
    );
    convert_vec_to_numpy(py, array)
}

fn convert_gradient_to_numpy(
    py: Python,
    input: Vec<Vec<(f32, f32)>>,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let (x_vals, y_vals): (Vec<f32>, Vec<f32>) = input
        .iter()
        .flat_map(|row: &Vec<(f32, f32)>| row.iter().map(|&(x, y)| (x, y)))
        .unzip();

    let rows: usize = input.len();
    let cols: usize = if rows > 0 { input[0].len() } else { 0 };

    let x_array2: Array2<f32> =
        Array2::from_shape_vec((rows, cols), x_vals).map_err(|e: numpy::ndarray::ShapeError| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create x array: {:?}",
                e
            ))
        })?;

    let y_array2: Array2<f32> =
        Array2::from_shape_vec((rows, cols), y_vals).map_err(|e: numpy::ndarray::ShapeError| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create y array: {:?}",
                e
            ))
        })?;

    let x_numpy_array: Bound<'_, PyArray<f32, Dim<[usize; 2]>>> = x_array2.into_pyarray(py);
    let y_numpy_array: Bound<'_, PyArray<f32, Dim<[usize; 2]>>> = y_array2.into_pyarray(py);

    Ok((
        x_numpy_array.to_owned().unbind(),
        y_numpy_array.to_owned().unbind(),
    ))
}

fn convert_vec_to_numpy(py: Python, input: Vec<Vec<f32>>) -> PyResult<Py<PyArray2<f32>>> {
    let array2: Array2<f32> = Array2::from_shape_vec((input.len(), input[0].len()), input.concat())
        .map_err(|e: numpy::ndarray::ShapeError| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create array: {:?}",
                e
            ))
        })?;

    let numpy_array: Bound<'_, PyArray<f32, Dim<[usize; 2]>>> = array2.into_pyarray(py);
    let numpy_array_bound: Py<PyArray<f32, Dim<[usize; 2]>>> = numpy_array.unbind();
    Ok(numpy_array_bound)
}

#[pymodule]
fn perlin_array(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_function(wrap_pyfunction!(get_perlin_numpy, m)?);
    let _ = m.add_function(wrap_pyfunction!(get_perlin_grid_numpy, m)?);
    let _ = m.add_function(wrap_pyfunction!(get_perlin_from_grid_numpy, m)?); 
    Ok(())
}

#[pyfunction]
fn get_perlin_grid_numpy(
    py: Python,
    grid_size: usize,
    octaves: usize,
    seed: u32,
    circular: bool,
    axis: &str,
) -> PyResult<Py<PyList>> {
    let grids: Vec<Vec<Vec<(f32, f32)>>> = create_grids(octaves, grid_size, seed, circular, axis);
    let mut numpy_grids: Vec<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> = Vec::new();
    for grid in grids.iter() {
        numpy_grids.push(convert_gradient_to_numpy(py, grid.clone())?);
    }

    let grids_list: Bound<'_, PyList> = PyList::new(py, numpy_grids)?;
    Ok(grids_list.unbind())
}

#[pyfunction]
fn get_perlin_from_grid_numpy(
    py: Python,
    grid: Py<PyList>,
    octaves: Py<PyList>,
    seed: u32,
    circular: bool,
    axis: &str,
    width: u32,
    height: u32,
) -> PyResult<Py<PyArray2<f32>>> {
    let octaves: Vec<f64> = octaves.extract(py)?;
    let grids: Vec<Vec<Vec<(f32, f32)>>> = convert_numpy_to_vec(py, grid)?;
    let length_grids: usize = grids.len();

    let array: Vec<Vec<f32>> = get_perlin_array(
        &Some(grids),
        length_grids,
        width,
        height,
        octaves,
        circular,
        axis,
        seed,
    );
    convert_vec_to_numpy(py, array)
}

fn convert_numpy_to_vec(py: Python, pylist: Py<PyList>) -> PyResult<Vec<Vec<Vec<(f32, f32)>>>> {
    let mut result: Vec<Vec<Vec<(f32, f32)>>> = Vec::new();
    for item in pylist.into_bound(py).iter() {
        let tuple: &Bound<'_, PyTuple> = item.downcast::<PyTuple>()?;
        let array1: numpy::ndarray::ArrayBase<numpy::ndarray::OwnedRepr<f32>, Dim<[usize; 2]>> =
            tuple
                .get_item(0)?
                .downcast::<PyArray2<f32>>()?
                .to_owned_array();
        let array2: numpy::ndarray::ArrayBase<numpy::ndarray::OwnedRepr<f32>, Dim<[usize; 2]>> =
            tuple
                .get_item(1)?
                .downcast::<PyArray2<f32>>()?
                .to_owned_array();

        let paired_data: Vec<Vec<(f32, f32)>> = array1
            .outer_iter()
            .zip(array2.outer_iter())
            .map(|(row1, row2)| {
                row1.iter()
                    .zip(row2.iter())
                    .map(|(&v1, &v2)| (v1, v2))
                    .collect::<Vec<(f32, f32)>>()
            })
            .collect();

        result.push(paired_data);
    }

    Ok(result)
}
