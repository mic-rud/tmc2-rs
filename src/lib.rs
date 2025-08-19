mod bitstream;
pub mod codec;
mod common;
mod decoder;
pub mod writer;

use bitstream::Bitstream;
use codec::PointSet3;
use common::context::Context;
use crossbeam_channel as chan;
use std::sync::mpsc::{channel, Receiver};
use std::path::PathBuf;
use std::thread;
use std::fs::File;
use std::io::Write;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

#[derive(Debug, Clone)]
pub enum BitstreamSource {
    File(PathBuf),
    Memory(Vec<u8>),
}

impl Default for BitstreamSource {
    fn default() -> Self {
        BitstreamSource::Memory(Vec::new())
    }
}


/// The library's decoder
pub struct Decoder {
    params: Params,
    // will be None once the decoder is started.
    tx: Option<chan::Sender<PointSet3>>,
    rx: chan::Receiver<PointSet3>,
}

/// Params to pass in to the library's decoder
#[derive(Debug, Default, Clone)]
pub struct Params {
    // NOTE: we don't need start_frame and reconstructed_data_path while decoding
    // pub start_frame: usize,
    // pub reconstructed_data_path: PathBuf,
    pub source: BitstreamSource,
    pub compressed_stream_path: PathBuf,
    pub video_decoder_path: Option<PathBuf>,
    // NOTE (2Jan23): always true
    // pub is_bytestream_video_coder: bool,
    pub keep_intermediate_files: bool,

    pub patch_color_subsampling: bool,
    pub color_space_conversion_path: Option<PathBuf>,
    pub inverse_color_space_conversion_config: Option<PathBuf>,

    // reconstruction options
    // NOTE (9Dec22): all set to default (false) for now since we are only supporting Rec0
    pixel_deinterleaving_type: bool,
    point_local_reconstruction_type: bool,
    reconstruction_eom_type: bool,
    _duplicated_point_removal_type: bool,
    reconstruct_raw_type: bool,
    apply_geo_smoothing_type: bool,
    apply_attr_smoothing_type: bool,
    attr_transfer_filter_type: bool,
    apply_occupancy_synthesis_type: bool,
}

impl Params {
    pub fn new(compressed_stream_path: PathBuf) -> Self {
        Self {
            compressed_stream_path,
            ..Default::default()
        }
    }

    // pub fn with_start_frame(mut self, start_frame: usize) -> Self {
    //     self.start_frame = start_frame;
    //     self
    // }

    // pub fn with_video_decoder(mut self, video_decoder_path: PathBuf) -> Self {
    //     self.video_decoder_path = Some(video_decoder_path);
    //     self
    // }
}

impl Decoder {
    pub fn new(params: Params) -> Self {
        let (tx, rx) = chan::bounded(1);
        Self {
            params,
            tx: Some(tx),
            rx,
        }
    }

    pub fn from_memory(data: Vec<u8>) -> Self {
        let params = Params {
            source: BitstreamSource::Memory(data),
            ..Default::default()
        };
        Decoder::new(params)
    }

    /// Spawns a thread to decode.
    /// The decoded point cloud can be retrieved in order by repeatedly calling `recv_frame()` method until it returns None.
    ///
    /// Caller needs to ensure that this function is only called once per Decoder instance. Calling more than once will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::PathBuf;
    /// use tmc2rs::{Decoder, Params};
    ///
    /// let mut decoder = Decoder::new(Params::new(PathBuf::from("path/to/compressed_stream")));
    /// decoder.start();
    /// for frame in decoder.into_iter() {
    ///    // do something with the frame
    /// }
    /// ```
    pub fn start(&mut self) {
        let bitstream = match &self.params.source {
            BitstreamSource::File(path) => Bitstream::from_file(path),
            BitstreamSource::Memory(data) => Bitstream::from_bytes(data.clone()),
        };
        // let mut bitstream_stat = bitstream::Stat::new();
        // TODO[checks] bitstream.computeMD5()
        // TODO[stat] (9Dec22): Do everything related to bitstream_stat
        // bitstream_stat.header = bitstream.size()
        let (mut ssvu, _header_size) =
            bitstream::reader::SampleStreamV3CUnit::from_bitstream(&bitstream);
        // TODO[stat] bitstream_stat.incr_header(header_size);

        let decoder = decoder::Decoder::new(self.params.clone());
        let tx = self
            .tx
            .take()
            .expect("library decoder can only be started once");

        thread::spawn(move || {
            // IDEA (9Dec22): We can parallelize iterations of this loop, since the data is self-contained.
            // i.e. AD, OVD, GVD, AVD are independent only of the VPS that immediately precedes it.
            // In the reference implementation, after running `ssvu.decode(...)`, the decoder is run, which kinda implies that there is some potential for parallelism here.
            // Check how `context.active_vps` is updated.
            while ssvu.get_v3c_unit_count() > 0 {
                // DIFF: This is different (I think) from the reference implementation.
                let mut context = Context::default();
                // TODO[stat] context.set_bitstream_stat(&bitstream_stat);
                ssvu.decode(&mut context);
                // TODO[checks]: context.check_profile()

                // context.atlas_contexts[i].allocate_video_frames(&mut context);
                // context.atlas_index = atl_id as u8;

                if let Err(_) = decoder.decode(&mut context, tx.clone()) {
                    // receiver `rx` dropped, so we can stop decoding.
                    break;
                }

                // SKIP: a bunch of if clauses on metrics.
            }

            drop(tx);
        });
    }

    /// Blocks the current thread until the next decoded frame is received.
    ///
    /// Once this method returns None, it will not block anymore as there are no more frames left to be decoded.
    pub fn recv_frame(&self) -> Option<PointSet3> {
        self.rx.recv().ok()
    }
}

impl Iterator for Decoder {
    type Item = PointSet3;

    fn next(&mut self) -> Option<Self::Item> {
        self.recv_frame()
    }
}


#[pyclass]
pub struct PyTMC2Decoder {
    decoder: Option<Decoder>,
}

#[pymethods]
impl PyTMC2Decoder {
    #[new]
    fn new(_py: Python<'_>, stream: &PyBytes) -> PyResult<Self> {
        let stream_data = stream.as_bytes().to_vec();

        let mut decoder = Decoder::from_memory(stream_data);

        decoder.start();

        Ok(Self {
            decoder: Some(decoder),
        })
    }

    fn next_frame(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if let Some(decoder) = &self.decoder {
            match decoder.recv_frame() {
                Some(frame) => {
                    let dict = PyDict::new(py);

                    let py_positions = PyList::empty(py);
                    for pos in frame.positions.iter() {
                        let tup = PyTuple::new(py, &[pos.x.into_py(py), pos.y.into_py(py), pos.z.into_py(py)]);
                        py_positions.append(tup).unwrap();
                    }
                    dict.set_item("positions", py_positions).ok();

                    if frame.with_colors {
                        let py_colors = PyList::empty(py);
                        for col in frame.colors.iter() {
                            let tup = PyTuple::new(py, &[col.x.into_py(py), col.y.into_py(py), col.z.into_py(py)]);
                            py_colors.append(tup).unwrap();
                        }
                        dict.set_item("colors", py_colors).ok();
                    }

                    Ok(Some(dict.into()))
                }
                None => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) {
        self.decoder = None;
    }
}

#[pymodule]
fn tmc2rs(_py: pyo3::Python, m: &pyo3::prelude::PyModule) -> pyo3::PyResult<()> {
    m.add_class::<PyTMC2Decoder>()?;
    Ok(())
}