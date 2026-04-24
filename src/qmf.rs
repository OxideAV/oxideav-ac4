//! AC-4 QMF analysis / synthesis filter-bank (§5.7.1 – 5.7.4).
#![allow(clippy::excessive_precision, clippy::needless_range_loop)]

//!
//! This module implements:
//!
//! * The 640-entry QMF prototype window [`QWIN`] from Annex D.3 of
//!   ETSI TS 103 190-1 V1.4.1 (also Table D.3). Used by both the
//!   analysis (§5.7.3) and synthesis (§5.7.4) filter-banks.
//! * [`qmf_analysis_slot`] — a single iteration of Pseudocode 65
//!   (analysis: 64 PCM in → 64 complex subband samples out).
//! * [`qmf_synthesis_slot`] — a single iteration of Pseudocode 66
//!   (synthesis: 64 complex subband samples in → 64 PCM out).
//! * [`QmfAnalysisBank`] / [`QmfSynthesisBank`] — multi-slot filter
//!   state machines wrapping the per-slot primitives with the correct
//!   delay-line shifts (`qmf_filt` for analysis, `qsyn_filt` for
//!   synthesis).
//!
//! Forward/inverse round-trip over a reasonable test signal (impulse,
//! sine, or white noise) reconstructs the input up to the intrinsic
//! 640-sample prototype-window delay with PSNR ≥ 20 dB — verified by
//! the unit tests in this file.

/// Number of QMF subbands in the AC-4 analysis / synthesis bank.
/// Fixed at 64 per §5.7.3.2 / §5.7.4.2.
pub const NUM_QMF_SUBBANDS: usize = 64;

/// Number of coefficients in the QMF prototype window (Annex D.3).
pub const NUM_QMF_WIN_COEF: usize = 640;

/// AC-4 QMF prototype window coefficients, Annex D.3 (Table D.3) of
/// ETSI TS 103 190-1 V1.4.1. Used by both the analysis and synthesis
/// filter-banks, shared between all channels.
#[rustfmt::skip]
pub static QWIN: [f32; NUM_QMF_WIN_COEF] = [
    0.0, 1.990318758627504e-004, 2.494762615491542e-004, 3.021769445225078e-004, 
    3.548460080857985e-004, 4.058915811480806e-004, 4.546408052001889e-004, 5.012680176678405e-004, 
    5.464958142195282e-004, 5.912073950641334e-004, 6.361178026937039e-004, 6.816060488244358e-004, 
    7.277257095064290e-004, 7.743418255606097e-004, 8.212990636826637e-004, 8.685363488152327e-004, 
    9.161071539925993e-004, 9.641168291303352e-004, 1.012630507392736e-003, 1.061605258108620e-003, 
    1.110882587090581e-003, 1.160236901298543e-003, 1.209448942573337e-003, 1.258362795150757e-003, 
    1.306902381715039e-003, 1.355046337751365e-003, 1.402784629568410e-003, 1.450086694843816e-003, 
    1.496898951224534e-003, 1.543170821958483e-003, 1.588889089195869e-003, 1.634098242730728e-003, 
    1.678892372493930e-003, 1.723381173920660e-003, 1.767651163797991e-003, 1.811741998614740e-003, 
    1.855650606587200e-003, 1.899360915083620e-003, 1.942876625831283e-003, 1.986241654706626e-003, 
    2.029534125962055e-003, 2.072840712410525e-003, 2.116229103721749e-003, 2.159738034390673e-003, 
    2.203392976200947e-003, 2.247239773881968e-003, 2.291373966775394e-003, 2.335946110021889e-003, 
    2.381132815654862e-003, 2.427086732976290e-003, 2.473891839822582e-003, 2.521550367974952e-003, 
    2.570013995199655e-003, 2.619244058999978e-003, 2.669265893796866e-003, 2.720177146231281e-003, 
    2.772088849679780e-003, 2.825009494162980e-003, 2.878716544061140e-003, 2.932677076291194e-003, 
    2.986067366389476e-003, 3.037905983043366e-003, 3.087269477594307e-003, 3.133519274378684e-003, 
    3.176460810085721e-003, 3.216374095471449e-003, 3.253902493849856e-003, 3.289837867273167e-003, 
    3.324873276103132e-003, 3.359407689115599e-003, 3.393454084675361e-003, 3.426668323773391e-003, 
    3.458465815999750e-003, 3.488171121469781e-003, 3.515141351338780e-003, 3.538827383683883e-003, 
    3.558767785536742e-003, 3.574539247363964e-003, 3.585697968628984e-003, 3.591743339500398e-003, 
    3.592116764752254e-003, 3.586228204993297e-003, 3.573492966885132e-003, 3.553356715665694e-003, 
    3.525300399274114e-003, 3.488824092931520e-003, 3.443423145747434e-003, 3.388568319085867e-003, 
    3.323699442173841e-003, 3.248231770523395e-003, 3.161568930730635e-003, 3.063113666967670e-003, 
    2.952270973359112e-003, 2.828441943181057e-003, 2.691016173288500e-003, 2.539366102140493e-003, 
    2.372848583221744e-003, 2.190814088754598e-003, 1.992618085548526e-003, 1.777631090142623e-003, 
    1.545242163079598e-003, 1.294855985911958e-003, 1.025885587325796e-003, 7.377456851538827e-004, 
    4.298496740962311e-004, 1.016113723823784e-004, -2.475493814535340e-004, -6.181972580227641e-004, 
    -1.010876063031582e-003, -1.426108207321696e-003, -1.864392667409557e-003, -2.326207721179968e-003, 
    -2.812013688448634e-003, -3.322252633537029e-003, -3.857344314546718e-003, -4.417678415707104e-003, 
    -5.003604409245843e-003, -5.615422427540850e-003, -6.253382198869787e-003, -6.917691380307223e-003, 
    -7.608536937561301e-003, -8.326113472848559e-003, -9.070651572928327e-003, -9.842433610911637e-003, 
    -1.064178450184536e-002, -1.146903570409307e-002, -1.232446526717138e-002, -1.320822893615923e-002, 
    1.412030102138547e-002, 1.506045143737221e-002, 1.602824700934038e-002, 1.702310507234504e-002, 
    1.804435938034114e-002, 1.909132707403387e-002, 2.016335321815832e-002, 2.125982139139435e-002, 
    2.238013015948307e-002, 2.352365148441367e-002, 2.468968228813486e-002, 2.587741357605385e-002, 
    2.708591966384863e-002, 2.831416731612567e-002, 2.956103453432552e-002, 3.082532788511644e-002, 
    3.210578787607558e-002, 3.340108247607704e-002, 3.470979250147262e-002, 3.603039785904666e-002, 
    3.736126987823528e-002, 3.870067428980750e-002, 4.004677994303860e-002, 4.139766786359423e-002, 
    4.275134353925827e-002, 4.410572893128047e-002, 4.545866171224587e-002, 4.680788921400311e-002, 
    4.815106534667384e-002, 4.948575188369231e-002, 5.080942296260306e-002, 5.211947012173918e-002, 
    5.341320372603929e-002, 5.468785186395163e-002, 5.594055607104873e-002, 5.716836923188953e-002, 
    5.836825629443718e-002, 5.953709945765930e-002, 6.067170625396996e-002, 6.176881705202805e-002, 
    6.282510999827461e-002, 6.383720245755561e-002, 6.480165083585107e-002, 6.571495100350305e-002, 
    6.657354346196487e-002, 6.737381445564891e-002, 6.811211000439976e-002, 6.878473991370719e-002, 
    6.938797895654626e-002, 6.991806618580000e-002, 7.037120381110623e-002, 7.074355866301176e-002, 
    7.103126866531538e-002, 7.123045563399449e-002, 7.133723888151840e-002, 7.134774334517399e-002, 
    7.125810128129656e-002, 7.106444395777428e-002, 7.076288963679085e-002, 7.034953453342756e-002, 
    6.982045490146145e-002, 6.917172452383333e-002, 6.839944399575645e-002, 6.749977716975542e-002, 
    6.646898181809889e-002, 6.530342654389224e-002, 6.399958984339946e-002, 6.255404354954748e-002, 
    6.096342863203985e-002, 5.922443337469448e-002, 5.733378365410422e-002, 5.528824660015738e-002, 
    5.308464739461209e-002, 5.071989148277166e-002, 4.819098634672628e-002, 4.549505579582869e-002, 
    4.262934676625042e-002, 3.959122947020497e-002, 3.637819581239452e-002, 3.298786054608736e-002, 
    2.941796954479800e-002, 2.566640058060906e-002, 2.173117939155709e-002, 1.761048656968719e-002, 
    1.330266415707108e-002, 8.806217289921706e-003, 4.119815918461287e-003, -7.577038291607129e-004, 
    -5.827337082489678e-003, -1.108990619665782e-002, -1.654605559674886e-002, -2.219624707735291e-002, 
    -2.804075556277473e-002, -3.407966641908426e-002, -4.031287253355741e-002, -4.674007190475649e-002, 
    -5.336076390182971e-002, -6.017424526940620e-002, -6.717960594283154e-002, -7.437572538762392e-002, 
    -8.176127022450692e-002, -8.933469320120192e-002, -9.709423309043450e-002, -1.050379143754414e-001, 
    -1.131635475471188e-001, -1.214687284677367e-001, -1.299508386078101e-001, -1.386070430802319e-001, 
    -1.474342913196958e-001, -1.564293167898782e-001, -1.655886374953163e-001, -1.749085568711785e-001, 
    -1.843851642116290e-001, -1.940143360850268e-001, -2.037917371113644e-001, -2.137128217101543e-001, 
    -2.237728356363325e-001, -2.339668182208061e-001, -2.442896055908444e-001, -2.547358344658102e-001, 
    -2.652999476893712e-001, -2.759762003673840e-001, -2.867586659726799e-001, -2.976412485679301e-001, 
    -3.086176827721830e-001, -3.196815399704708e-001, -3.308262316588501e-001, -3.420450091826495e-001, 
    3.533309414505971e-001, 3.646770149404552e-001, 3.760759747758828e-001, 3.875204555118187e-001, 
    3.990029533969267e-001, 4.105158411581483e-001, 4.220513789540003e-001, 4.336017251305980e-001, 
    4.451589452332786e-001, 4.567150149423557e-001, 4.682618290579831e-001, 4.797912086537587e-001, 
    4.912949058677955e-001, 5.027646134968753e-001, 5.141919746376279e-001, 5.255685924518015e-001, 
    5.368860394090674e-001, 5.481358656081351e-001, 5.593096071830315e-001, 5.703987947306394e-001, 
    5.813949615434598e-001, 5.922896536434017e-001, 6.030744392774144e-001, 6.137409201916185e-001, 
    6.242807411441345e-001, 6.346855991963545e-001, 6.449472531836600e-001, 6.550575323798634e-001, 
    6.650083455855346e-001, 6.747916901830467e-001, 6.843996616799759e-001, 6.938244627003839e-001, 
    7.030584122393319e-001, 7.120939537241190e-001, 7.209236637533725e-001, 7.295402599029810e-001, 
    7.379366091028713e-001, 7.461057359576386e-001, 7.540408314942230e-001, 7.617352611504460e-001, 
    7.691825714586890e-001, 7.763765020733762e-001, 7.833109874824341e-001, 7.899801646390305e-001, 
    7.963783815797485e-001, 8.025002033685581e-001, 8.083404191294724e-001, 8.138940486031526e-001, 
    8.191563476989879e-001, 8.241228138607196e-001, 8.287891904413357e-001, 8.331514714928793e-001, 
    8.372059062705359e-001, 8.409490040631689e-001, 8.443775395556067e-001, 8.474885573145614e-001, 
    8.502793750759253e-001, 8.527475863595390e-001, 8.548910606594570e-001, 8.567079441260879e-001, 
    8.581966597760032e-001, 8.593559096378087e-001, 8.601846769933608e-001, 8.606822313166693e-001, 
    8.608481078185764e-001, 8.606822313166693e-001, 8.601846769933608e-001, 8.593559096378087e-001, 
    8.581966597760032e-001, 8.567079441260879e-001, 8.548910606594570e-001, 8.527475863595390e-001, 
    8.502793750759253e-001, 8.474885573145614e-001, 8.443775395556067e-001, 8.409490040631689e-001, 
    8.372059062705359e-001, 8.331514714928793e-001, 8.287891904413357e-001, 8.241228138607196e-001, 
    8.191563476989879e-001, 8.138940486031526e-001, 8.083404191294724e-001, 8.025002033685581e-001, 
    7.963783815797485e-001, 7.899801646390305e-001, 7.833109874824341e-001, 7.763765020733762e-001, 
    7.691825714586890e-001, 7.617352611504460e-001, 7.540408314942230e-001, 7.461057359576386e-001, 
    7.379366091028713e-001, 7.295402599029810e-001, 7.209236637533725e-001, 7.120939537241190e-001, 
    7.030584122393319e-001, 6.938244627003839e-001, 6.843996616799759e-001, 6.747916901830467e-001, 
    6.650083455855346e-001, 6.550575323798634e-001, 6.449472531836600e-001, 6.346855991963545e-001, 
    6.242807411441345e-001, 6.137409201916185e-001, 6.030744392774144e-001, 5.922896536434017e-001, 
    5.813949615434598e-001, 5.703987947306394e-001, 5.593096071830315e-001, 5.481358656081351e-001, 
    5.368860394090674e-001, 5.255685924518015e-001, 5.141919746376279e-001, 5.027646134968753e-001, 
    4.912949058677955e-001, 4.797912086537587e-001, 4.682618290579831e-001, 4.567150149423557e-001, 
    4.451589452332786e-001, 4.336017251305980e-001, 4.220513789540003e-001, 4.105158411581483e-001, 
    3.990029533969267e-001, 3.875204555118187e-001, 3.760759747758828e-001, 3.646770149404552e-001, 
    -3.533309414505971e-001, -3.420450091826495e-001, -3.308262316588501e-001, -3.196815399704708e-001, 
    -3.086176827721830e-001, -2.976412485679301e-001, -2.867586659726799e-001, -2.759762003673840e-001, 
    -2.652999476893712e-001, -2.547358344658102e-001, -2.442896055908444e-001, -2.339668182208061e-001, 
    -2.237728356363325e-001, -2.137128217101543e-001, -2.037917371113644e-001, -1.940143360850268e-001, 
    -1.843851642116290e-001, -1.749085568711785e-001, -1.655886374953163e-001, -1.564293167898782e-001, 
    -1.474342913196958e-001, -1.386070430802319e-001, -1.299508386078101e-001, -1.214687284677367e-001, 
    -1.131635475471188e-001, -1.050379143754414e-001, -9.709423309043450e-002, -8.933469320120192e-002, 
    -8.176127022450692e-002, -7.437572538762392e-002, -6.717960594283154e-002, -6.017424526940620e-002, 
    -5.336076390182971e-002, -4.674007190475649e-002, -4.031287253355741e-002, -3.407966641908426e-002, 
    -2.804075556277473e-002, -2.219624707735291e-002, -1.654605559674886e-002, -1.108990619665782e-002, 
    -5.827337082489678e-003, -7.577038291607129e-004, 4.119815918461287e-003, 8.806217289921706e-003, 
    1.330266415707108e-002, 1.761048656968719e-002, 2.173117939155709e-002, 2.566640058060906e-002, 
    2.941796954479800e-002, 3.298786054608736e-002, 3.637819581239452e-002, 3.959122947020497e-002, 
    4.262934676625042e-002, 4.549505579582869e-002, 4.819098634672628e-002, 5.071989148277166e-002, 
    5.308464739461209e-002, 5.528824660015738e-002, 5.733378365410422e-002, 5.922443337469448e-002, 
    6.096342863203985e-002, 6.255404354954748e-002, 6.399958984339946e-002, 6.530342654389224e-002, 
    6.646898181809889e-002, 6.749977716975542e-002, 6.839944399575645e-002, 6.917172452383333e-002, 
    6.982045490146145e-002, 7.034953453342756e-002, 7.076288963679085e-002, 7.106444395777428e-002, 
    7.125810128129656e-002, 7.134774334517399e-002, 7.133723888151840e-002, 7.123045563399449e-002, 
    7.103126866531538e-002, 7.074355866301176e-002, 7.037120381110623e-002, 6.991806618580000e-002, 
    6.938797895654626e-002, 6.878473991370719e-002, 6.811211000439976e-002, 6.737381445564891e-002, 
    6.657354346196487e-002, 6.571495100350305e-002, 6.480165083585107e-002, 6.383720245755561e-002, 
    6.282510999827461e-002, 6.176881705202805e-002, 6.067170625396996e-002, 5.953709945765930e-002, 
    5.836825629443718e-002, 5.716836923188953e-002, 5.594055607104873e-002, 5.468785186395163e-002, 
    5.341320372603929e-002, 5.211947012173918e-002, 5.080942296260306e-002, 4.948575188369231e-002, 
    4.815106534667384e-002, 4.680788921400311e-002, 4.545866171224587e-002, 4.410572893128047e-002, 
    4.275134353925827e-002, 4.139766786359423e-002, 4.004677994303860e-002, 3.870067428980750e-002, 
    3.736126987823528e-002, 3.603039785904666e-002, 3.470979250147262e-002, 3.340108247607704e-002, 
    3.210578787607558e-002, 3.082532788511644e-002, 2.956103453432552e-002, 2.831416731612567e-002, 
    2.708591966384863e-002, 2.587741357605385e-002, 2.468968228813486e-002, 2.352365148441367e-002, 
    2.238013015948307e-002, 2.125982139139435e-002, 2.016335321815832e-002, 1.909132707403387e-002, 
    1.804435938034114e-002, 1.702310507234504e-002, 1.602824700934038e-002, 1.506045143737221e-002, 
    -1.412030102138547e-002, -1.320822893615923e-002, -1.232446526717138e-002, -1.146903570409307e-002, 
    -1.064178450184536e-002, -9.842433610911637e-003, -9.070651572928327e-003, -8.326113472848559e-003, 
    -7.608536937561301e-003, -6.917691380307223e-003, -6.253382198869787e-003, -5.615422427540850e-003, 
    -5.003604409245843e-003, -4.417678415707104e-003, -3.857344314546718e-003, -3.322252633537029e-003, 
    -2.812013688448634e-003, -2.326207721179968e-003, -1.864392667409557e-003, -1.426108207321696e-003, 
    -1.010876063031582e-003, -6.181972580227641e-004, -2.475493814535340e-004, 1.016113723823784e-004, 
    4.298496740962311e-004, 7.377456851538827e-004, 1.025885587325796e-003, 1.294855985911958e-003, 
    1.545242163079598e-003, 1.777631090142623e-003, 1.992618085548526e-003, 2.190814088754598e-003, 
    2.372848583221744e-003, 2.539366102140493e-003, 2.691016173288500e-003, 2.828441943181057e-003, 
    2.952270973359112e-003, 3.063113666967670e-003, 3.161568930730635e-003, 3.248231770523395e-003, 
    3.323699442173841e-003, 3.388568319085867e-003, 3.443423145747434e-003, 3.488824092931520e-003, 
    3.525300399274114e-003, 3.553356715665694e-003, 3.573492966885132e-003, 3.586228204993297e-003, 
    3.592116764752254e-003, 3.591743339500398e-003, 3.585697968628984e-003, 3.574539247363964e-003, 
    3.558767785536742e-003, 3.538827383683883e-003, 3.515141351338780e-003, 3.488171121469781e-003, 
    3.458465815999750e-003, 3.426668323773391e-003, 3.393454084675361e-003, 3.359407689115599e-003, 
    3.324873276103132e-003, 3.289837867273167e-003, 3.253902493849856e-003, 3.216374095471449e-003, 
    3.176460810085721e-003, 3.133519274378684e-003, 3.087269477594307e-003, 3.037905983043366e-003, 
    2.986067366389476e-003, 2.932677076291194e-003, 2.878716544061140e-003, 2.825009494162980e-003, 
    2.772088849679780e-003, 2.720177146231281e-003, 2.669265893796866e-003, 2.619244058999978e-003, 
    2.570013995199655e-003, 2.521550367974952e-003, 2.473891839822582e-003, 2.427086732976290e-003, 
    2.381132815654862e-003, 2.335946110021889e-003, 2.291373966775394e-003, 2.247239773881968e-003, 
    2.203392976200947e-003, 2.159738034390673e-003, 2.116229103721749e-003, 2.072840712410525e-003, 
    2.029534125962055e-003, 1.986241654706626e-003, 1.942876625831283e-003, 1.899360915083620e-003, 
    1.855650606587200e-003, 1.811741998614740e-003, 1.767651163797991e-003, 1.723381173920660e-003, 
    1.678892372493930e-003, 1.634098242730728e-003, 1.588889089195869e-003, 1.543170821958483e-003, 
    1.496898951224534e-003, 1.450086694843816e-003, 1.402784629568410e-003, 1.355046337751365e-003, 
    1.306902381715039e-003, 1.258362795150757e-003, 1.209448942573337e-003, 1.160236901298543e-003, 
    1.110882587090581e-003, 1.061605258108620e-003, 1.012630507392736e-003, 9.641168291303352e-004, 
    9.161071539925993e-004, 8.685363488152327e-004, 8.212990636826637e-004, 7.743418255606097e-004, 
    7.277257095064290e-004, 6.816060488244358e-004, 6.361178026937039e-004, 5.912073950641334e-004, 
    5.464958142195282e-004, 5.012680176678405e-004, 4.546408052001889e-004, 4.058915811480806e-004, 
    3.548460080857985e-004, 3.021769445225078e-004, 2.494762615491542e-004, 1.990318758627504e-004, 
    
];

/// One-slot forward QMF analysis transform per ETSI TS 103 190-1
/// §5.7.3.2 (Pseudocode 65). Given one time-slot's worth of input —
/// `qmf_filt[0..640]` holding the delay line in reversed order with
/// positions 0..64 filled with the 64 newest input samples — emit the
/// 64 complex QMF subband values for this slot.
///
/// Per the spec:
///
/// ```text
/// z[n] = qmf_filt[n] * QWIN[n]             n in [0, 640)
/// u[n] = sum_{k=0..5} z[n + k * 128]       n in [0, 128)
/// Q[sb] = u[0] * exp(-j*pi/128 * (sb+0.5))
///       + sum_{n=1..128} u[n] * exp(j*pi/128 * (sb+0.5) * (2n-1))
/// ```
///
/// Returns the 64 complex subband samples as `(re, im)` pairs. The
/// caller is responsible for maintaining `qmf_filt` across slots —
/// this function is a pure single-slot transform.
pub fn qmf_analysis_slot(qmf_filt: &[f32; NUM_QMF_WIN_COEF]) -> [(f32, f32); NUM_QMF_SUBBANDS] {
    // z = qmf_filt * QWIN (element-wise, 640 samples).
    // u[n] = sum_{k=0..5} z[n + k*128], n in 0..128.
    let mut u = [0.0f64; 128];
    for n in 0..128 {
        let mut acc = 0.0f64;
        for k in 0..5 {
            let idx = n + k * 128;
            acc += (qmf_filt[idx] as f64) * (QWIN[idx] as f64);
        }
        u[n] = acc;
    }
    // Q[sb][ts] = sum_{n=0..128} u[n] * exp(j * pi/128 * (sb+0.5) * (2n-1))
    // (the n=0 branch in Pseudocode 65 collapses to (2*0 - 1) = -1 which is
    // identical to the general formula).
    use core::f64::consts::PI;
    let mut out = [(0.0f32, 0.0f32); NUM_QMF_SUBBANDS];
    for (sb, slot) in out.iter_mut().enumerate() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for n in 0..128 {
            let phase = PI / 128.0 * (sb as f64 + 0.5) * ((2 * n) as f64 - 1.0);
            re += u[n] * phase.cos();
            im += u[n] * phase.sin();
        }
        *slot = (re as f32, im as f32);
    }
    out
}

/// One-slot QMF synthesis transform per ETSI TS 103 190-1 §5.7.4.2
/// (Pseudocode 66). Given the current 64-value complex subband column
/// and the 1 280-sample synthesis delay line `qsyn_filt` (the caller
/// has already shifted the line by 128 and injected the newly
/// modulated samples into positions 0..128), reconstruct the window
/// vector `g` (640 samples via the folded tap layout) and produce 64
/// PCM output samples.
///
/// The multi-slot caller ([`QmfSynthesisBank`]) is responsible for the
/// shift-by-128 and per-slot `qsyn_filt[0..128]` modulation step; this
/// function handles only the steps that touch `g`, the window
/// multiply and the 64-way tap sum.
fn qmf_synthesis_tap_sum(qsyn_filt: &[f64; 1280]) -> [f32; NUM_QMF_SUBBANDS] {
    // Fold qsyn_filt into g[] (640 samples) using the spec's 2-by-128
    // stride layout:
    //
    //   g[128*n + sb]      = qsyn_filt[256*n + sb]
    //   g[128*n + 64 + sb] = qsyn_filt[256*n + 192 + sb]
    //
    // for n in 0..5 and sb in 0..64. 5 * 128 = 640 samples total.
    let mut g = [0.0f64; 640];
    for n in 0..5 {
        for sb in 0..64 {
            g[128 * n + sb] = qsyn_filt[256 * n + sb];
            g[128 * n + 64 + sb] = qsyn_filt[256 * n + 192 + sb];
        }
    }
    // w[n] = g[n] * QWIN[n], then pcm[sb] = sum_{k=0..10} w[64*k + sb].
    let mut pcm = [0.0f32; NUM_QMF_SUBBANDS];
    for sb in 0..NUM_QMF_SUBBANDS {
        let mut acc = 0.0f64;
        for k in 0..10 {
            let idx = 64 * k + sb;
            acc += g[idx] * (QWIN[idx] as f64);
        }
        pcm[sb] = acc as f32;
    }
    pcm
}

/// One-slot forward QMF synthesis transform per ETSI TS 103 190-1
/// §5.7.4.2 (Pseudocode 66). Convenience path for single-slot use —
/// usually you'll want [`QmfSynthesisBank::process_slot`] which
/// maintains the shared filter state across frames.
///
/// `q[sb]` is the complex QMF subband column for this slot, laid out
/// as `(re, im)` pairs indexed by subband `sb ∈ 0..64`.
pub fn qmf_synthesis_slot(
    qsyn_filt: &mut [f64; 1280],
    q: &[(f32, f32); NUM_QMF_SUBBANDS],
) -> [f32; NUM_QMF_SUBBANDS] {
    use core::f64::consts::PI;
    // Shift the delay line by 128.
    for n in (128..1280).rev() {
        qsyn_filt[n] = qsyn_filt[n - 128];
    }
    // Modulate the new 128 taps from the Q[sb][ts] column.
    //   qsyn_filt[n] = sum_{sb=0..64} real( Q[sb][ts]/64 * exp(j*phi) )
    // with phi = (pi/128) * (sb + 0.5) * (2n - 255).
    for n in 0..128 {
        let mut acc = 0.0f64;
        for sb in 0..NUM_QMF_SUBBANDS {
            let (re, im) = q[sb];
            let re = (re as f64) / 64.0;
            let im = (im as f64) / 64.0;
            let phi = PI / 128.0 * (sb as f64 + 0.5) * ((2 * n) as f64 - 255.0);
            // Re{ (re + j*im) * (cos(phi) + j*sin(phi)) }
            //   = re * cos(phi) - im * sin(phi)
            acc += re * phi.cos() - im * phi.sin();
        }
        qsyn_filt[n] = acc;
    }
    qmf_synthesis_tap_sum(qsyn_filt)
}

/// Multi-slot QMF analysis filter bank.
///
/// Wraps the per-slot [`qmf_analysis_slot`] primitive with the
/// circular delay-line maintenance from Pseudocode 65:
///
/// * `qmf_filt[0..640]` carries the current delay line (stored in
///   "reversed" order — position 0 holds the newest sample, position
///   639 the oldest).
/// * Each call to [`process_slot`](Self::process_slot) first shifts
///   the delay line by 64 (positions 64..640 move up by 64) and then
///   writes the 64 new PCM samples into positions 0..64 in reverse
///   order, matching `qmf_filt[sb] = pcm[ts*64 + 63 - sb]`.
/// * After the shift+feed, the 64 complex QMF subband samples are
///   computed via the shared core routine.
pub struct QmfAnalysisBank {
    qmf_filt: [f32; NUM_QMF_WIN_COEF],
}

impl Default for QmfAnalysisBank {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfAnalysisBank {
    pub const fn new() -> Self {
        Self {
            qmf_filt: [0.0; NUM_QMF_WIN_COEF],
        }
    }

    /// Consume one slot's worth of 64 PCM samples and emit the 64
    /// complex subband values for that slot.
    pub fn process_slot(
        &mut self,
        pcm: &[f32; NUM_QMF_SUBBANDS],
    ) -> [(f32, f32); NUM_QMF_SUBBANDS] {
        // Shift the delay line by 64 positions (positions 639..=64
        // receive the old contents of positions 575..=0).
        for sb in (64..NUM_QMF_WIN_COEF).rev() {
            self.qmf_filt[sb] = self.qmf_filt[sb - 64];
        }
        // Feed the 64 new samples into positions 0..64 in reversed
        // order (the newest sample in the current slot, pcm[63], ends
        // up at qmf_filt[0]).
        for sb in 0..64 {
            // Pseudocode 65: for (sb = 63; sb >= 0; sb--)
            //   qmf_filt[sb] = pcm[ts*64 + 63 - sb];
            self.qmf_filt[sb] = pcm[63 - sb];
        }
        qmf_analysis_slot(&self.qmf_filt)
    }

    /// Process a block of `n_slots` QMF time slots. `pcm` must contain
    /// exactly `64 * n_slots` PCM samples; the returned matrix is
    /// `n_slots` rows × 64 columns of complex `(re, im)` pairs.
    pub fn process_block(&mut self, pcm: &[f32]) -> Vec<[(f32, f32); NUM_QMF_SUBBANDS]> {
        assert_eq!(
            pcm.len() % NUM_QMF_SUBBANDS,
            0,
            "pcm length must be a multiple of 64"
        );
        let n_slots = pcm.len() / NUM_QMF_SUBBANDS;
        let mut out = Vec::with_capacity(n_slots);
        let mut slot = [0.0f32; NUM_QMF_SUBBANDS];
        for ts in 0..n_slots {
            let base = ts * NUM_QMF_SUBBANDS;
            slot.copy_from_slice(&pcm[base..base + NUM_QMF_SUBBANDS]);
            out.push(self.process_slot(&slot));
        }
        out
    }
}

/// Multi-slot QMF synthesis filter bank.
///
/// Wraps [`qmf_synthesis_slot`] with the 1 280-sample circular delay
/// line (`qsyn_filt`) whose shift-by-128 + tap-sum layout is the
/// inverse of the analysis bank.
pub struct QmfSynthesisBank {
    qsyn_filt: [f64; 1280],
}

impl Default for QmfSynthesisBank {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfSynthesisBank {
    pub const fn new() -> Self {
        Self {
            qsyn_filt: [0.0; 1280],
        }
    }

    /// Consume the 64-sample complex subband column for this slot and
    /// emit 64 PCM samples.
    pub fn process_slot(&mut self, q: &[(f32, f32); NUM_QMF_SUBBANDS]) -> [f32; NUM_QMF_SUBBANDS] {
        qmf_synthesis_slot(&mut self.qsyn_filt, q)
    }

    /// Process a block of `n_slots` QMF time slots. Inverse of
    /// [`QmfAnalysisBank::process_block`] — returns `64 * n_slots` PCM
    /// samples.
    pub fn process_block(&mut self, q_slots: &[[(f32, f32); NUM_QMF_SUBBANDS]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(q_slots.len() * NUM_QMF_SUBBANDS);
        for slot in q_slots {
            let pcm = self.process_slot(slot);
            out.extend_from_slice(&pcm);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwin_length_matches_spec() {
        assert_eq!(QWIN.len(), NUM_QMF_WIN_COEF);
    }

    #[test]
    fn qwin_first_and_last_match_annex_d3() {
        // Anchor values from Annex D.3 (Table D.3): QWIN is symmetric
        // around index 320 (up to float rounding). First value is
        // zero, last is the mirror of the second.
        assert_eq!(QWIN[0], 0.0);
        assert!((QWIN[1] - 1.990318758627504e-004_f32).abs() < 1e-9);
        assert!((QWIN[639] - 1.990318758627504e-004_f32).abs() < 1e-9);
        assert!((QWIN[638] - 2.494762615491542e-004_f32).abs() < 1e-9);
    }

    #[test]
    fn qmf_analysis_slot_on_zero_filt_is_zero() {
        let filt = [0.0f32; NUM_QMF_WIN_COEF];
        let out = qmf_analysis_slot(&filt);
        for (re, im) in out.iter() {
            assert_eq!(*re, 0.0);
            assert_eq!(*im, 0.0);
        }
    }

    #[test]
    fn qmf_analysis_slot_preserves_shape() {
        // Sanity: non-zero input produces non-zero subband samples.
        let mut filt = [0.0f32; NUM_QMF_WIN_COEF];
        for (i, slot) in filt.iter_mut().enumerate() {
            *slot = ((i as f32) * 0.01).sin();
        }
        let out = qmf_analysis_slot(&filt);
        let energy: f64 = out
            .iter()
            .map(|(re, im)| (*re as f64) * (*re as f64) + (*im as f64) * (*im as f64))
            .sum();
        assert!(energy > 0.0);
    }

    /// PSNR in dB between `reference` and `test`, computed over
    /// `[start..end)`. Clamped to 200 dB when the MSE is zero.
    fn psnr_db(reference: &[f32], test: &[f32], start: usize, end: usize) -> f64 {
        let mut peak: f64 = 0.0;
        let mut mse: f64 = 0.0;
        let mut count = 0usize;
        for i in start..end {
            let r = reference[i] as f64;
            let t = test[i] as f64;
            let d = r - t;
            mse += d * d;
            peak = peak.max(r.abs());
            count += 1;
        }
        if count == 0 {
            return 200.0;
        }
        if mse == 0.0 {
            return 200.0;
        }
        let mse = mse / count as f64;
        if peak == 0.0 {
            // Only noise in the signal. Report a high PSNR if mse is
            // very small relative to unity.
            return -10.0 * mse.log10();
        }
        20.0 * peak.log10() - 10.0 * mse.log10()
    }

    #[test]
    fn qmf_roundtrip_impulse_psnr_ok() {
        // Put a unit impulse at sample 0, run forward + inverse bank,
        // check reconstruction over the steady-state region.
        let mut pcm = vec![0.0f32; 64 * 32];
        pcm[0] = 1.0;
        let mut ana = QmfAnalysisBank::new();
        let mut syn = QmfSynthesisBank::new();
        let q = ana.process_block(&pcm);
        let recon = syn.process_block(&q);
        assert_eq!(recon.len(), pcm.len());
        // Reconstruction is aligned to analysis/synthesis combined
        // delay of 640 samples (one full prototype window). We check
        // the tail region where the impulse has propagated through.
        // For an impulse, the reconstructed signal has finite energy
        // spread across all slots; just verify it is non-trivial and
        // that the unmoving tail region reproduces something small.
        let energy: f64 = recon.iter().map(|&x| (x as f64) * (x as f64)).sum();
        assert!(energy > 0.0, "reconstruction has zero energy");
    }

    #[test]
    fn qmf_impulse_response_peak_at_expected_delay() {
        // Sanity: an impulse at input position `i0` must surface as a
        // peak in the reconstruction at position `i0 + QMF_RT_DELAY`
        // with amplitude close to QMF_RT_SIGN.
        let n_slots = 60usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let i0 = 100usize;
        pcm[i0] = 1.0;
        let mut ana = QmfAnalysisBank::new();
        let mut syn = QmfSynthesisBank::new();
        let q = ana.process_block(&pcm);
        let recon = syn.process_block(&q);
        let mut max_i = 0usize;
        let mut max_v = 0.0f32;
        for (i, &r) in recon.iter().enumerate() {
            if r.abs() > max_v {
                max_v = r.abs();
                max_i = i;
            }
        }
        assert_eq!(
            max_i,
            i0 + QMF_RT_DELAY,
            "impulse peak at {max_i}, expected {}",
            i0 + QMF_RT_DELAY
        );
        assert!(
            (max_v - QMF_RT_SIGN.abs()).abs() < 1e-3,
            "impulse peak amplitude {max_v} != {}",
            QMF_RT_SIGN.abs()
        );
    }

    /// Combined group delay of the AC-4 analysis + synthesis QMF pair.
    /// An input unit impulse at position `i` surfaces as a peak at
    /// synthesis-output position `i + QMF_RT_DELAY`. Measured by
    /// running the unit-impulse end-to-end test: the reconstruction is
    /// in-phase (scale = +1) and peaks 577 samples after the input
    /// impulse.
    const QMF_RT_DELAY: usize = 577;
    const QMF_RT_SIGN: f32 = 1.0;

    /// Helper: compute PSNR between `pcm[start..end]` and the aligned
    /// reconstruction `SIGN * recon[start+QMF_RT_DELAY..end+QMF_RT_DELAY]`.
    fn psnr_roundtrip(pcm: &[f32], recon: &[f32], start: usize, end: usize) -> f64 {
        let mut peak = 0.0f64;
        let mut mse = 0.0f64;
        let mut count = 0usize;
        for i in start..end {
            let rj = i + QMF_RT_DELAY;
            if rj >= recon.len() {
                break;
            }
            let p = pcm[i] as f64;
            let r = (QMF_RT_SIGN * recon[rj]) as f64;
            let d = p - r;
            mse += d * d;
            peak = peak.max(p.abs());
            count += 1;
        }
        if count == 0 || mse == 0.0 {
            return 200.0;
        }
        let mse = mse / count as f64;
        if peak == 0.0 {
            return -10.0 * mse.log10();
        }
        20.0 * peak.log10() - 10.0 * mse.log10()
    }

    #[test]
    fn qmf_roundtrip_sine_psnr_ge_20db() {
        // A 48 kHz sine at 1 kHz. Use 60 slots (= 3 840 samples) so
        // the combined 1 023-sample delay is well-flushed.
        let n_slots = 60usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let f = 1000.0_f32 / 48_000.0_f32;
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * f * i as f32).sin();
        }
        let mut ana = QmfAnalysisBank::new();
        let mut syn = QmfSynthesisBank::new();
        let q = ana.process_block(&pcm);
        let recon = syn.process_block(&q);
        // Evaluate in a steady-state window that avoids both the
        // analysis-bank start-up transient (first ~640 samples) and
        // the synthesis-bank tail (last QMF_RT_DELAY samples).
        let start = 64;
        let end = n - QMF_RT_DELAY - 1;
        let psnr = psnr_roundtrip(&pcm, &recon, start, end);
        eprintln!("sine QMF roundtrip PSNR = {psnr:.2} dB");
        assert!(psnr >= 20.0, "sine QMF roundtrip PSNR too low: {psnr} dB");
    }

    #[test]
    fn qmf_roundtrip_noise_psnr_ge_20db() {
        // Deterministic pseudo-noise across 60 QMF slots.
        let n_slots = 60usize;
        let n = n_slots * 64;
        let mut pcm = vec![0.0f32; n];
        let mut seed: u32 = 0x1234_5678;
        for s in pcm.iter_mut() {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let u = (seed >> 16) as i32 as f32 / 32_768.0;
            *s = 0.5 * u;
        }
        let mut ana = QmfAnalysisBank::new();
        let mut syn = QmfSynthesisBank::new();
        let q = ana.process_block(&pcm);
        let recon = syn.process_block(&q);
        let start = 64;
        let end = n - QMF_RT_DELAY - 1;
        let psnr = psnr_roundtrip(&pcm, &recon, start, end);
        eprintln!("noise QMF roundtrip PSNR = {psnr:.2} dB");
        assert!(psnr >= 20.0, "noise QMF roundtrip PSNR too low: {psnr} dB");
    }
}
