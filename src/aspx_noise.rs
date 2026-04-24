//! A-SPX noise generator (§5.7.6.4.3) + Annex D.2 NoiseTable.
//!
//! This module implements the ETSI TS 103 190-1 V1.4.1 A-SPX noise
//! generator tool (§5.7.6.4.3, Pseudocodes 102/103) on top of the
//! 512-entry complex random noise table from Annex D.2 (table D.2).
//! The generator injects a band-shaped complex noise stream into the
//! HF part of the QMF matrix; downstream the HF signal assembler
//! (§5.7.6.4.5 Pseudocode 107) adds it to the regenerated tile
//! signal.
//!
//! The caller is expected to have already derived:
//! * The frequency tables (sbx / sbz / num_sb_aspx / num_sbg_noise)
//!   from §5.7.6.3.1.
//! * The signal envelope borders `atsg_sig[]` from Pseudocode 76
//!   (Table 194 for FIXFIX).
//! * The noise-level matrix `noise_lev_sb_adj[sb][atsg_noise]`
//!   (shape num_sb_aspx × num_atsg_noise) from the envelope-
//!   adjustment tool (§5.7.6.4.2 Pseudocode 94 / Pseudocode 101).
//!
//! The generator itself is purely deterministic: a counter-indexed
//! walk over the NoiseTable, mixed with the per-subband-per-envelope
//! level factor, written into the QMF high band.

#![allow(clippy::excessive_precision)]

/// Annex D.2 ASPX_NOISE table — 512 complex values with random phase
/// and average unit energy. ETSI TS 103 190-1 V1.4.1, table D.2.
///
/// Indexed `[index][0]` real part, `[index][1]` imaginary part. Here
/// we store tuples of `(re, im)` and expose a [`NOISE_TABLE_LEN`]
/// constant at 512 — this is the modulus used by Pseudocode 103's
/// `noise_idx` helper.
pub const NOISE_TABLE_LEN: usize = 512;

pub const ASPX_NOISE_TABLE: [(f32, f32); NOISE_TABLE_LEN] = [
    (-0.782083_f32, -0.623174_f32),
    (0.705088_f32, -0.70912_f32),
    (0.268786_f32, -0.9632_f32),
    (0.689305_f32, -0.724471_f32),
    (-0.0430946_f32, -0.999071_f32),
    (-0.998077_f32, -0.0619922_f32),
    (0.867875_f32, -0.496783_f32),
    (0.814907_f32, -0.579591_f32),
    (0.880168_f32, 0.474663_f32),
    (-0.39062_f32, -0.920552_f32),
    (0.0434465_f32, 0.999056_f32),
    (0.610173_f32, -0.792268_f32),
    (-0.942195_f32, 0.335065_f32),
    (-0.911161_f32, 0.412052_f32),
    (-0.999084_f32, -0.0427925_f32),
    (-0.811795_f32, 0.583942_f32),
    (0.836524_f32, 0.54793_f32),
    (0.475443_f32, 0.879747_f32),
    (0.961597_f32, -0.274466_f32),
    (0.820146_f32, 0.572155_f32),
    (-0.498318_f32, -0.866994_f32),
    (0.825586_f32, -0.564277_f32),
    (0.999973_f32, 0.00730176_f32),
    (0.895923_f32, 0.44421_f32),
    (-0.80517_f32, -0.593044_f32),
    (0.277753_f32, 0.960652_f32),
    (-0.999688_f32, 0.0249582_f32),
    (-0.802608_f32, -0.596507_f32),
    (0.936737_f32, -0.350035_f32),
    (-0.997477_f32, -0.0709933_f32),
    (-0.674713_f32, -0.73808_f32),
    (-0.957686_f32, -0.287817_f32),
    (0.983532_f32, 0.180732_f32),
    (0.634759_f32, -0.77271_f32),
    (-0.150723_f32, -0.988576_f32),
    (-0.113979_f32, 0.993483_f32),
    (-0.993827_f32, -0.110942_f32),
    (0.781536_f32, 0.62386_f32),
    (-0.615104_f32, -0.788446_f32),
    (-0.586834_f32, 0.809707_f32),
    (-0.226253_f32, 0.974069_f32),
    (0.949896_f32, -0.312565_f32),
    (0.830084_f32, 0.557638_f32),
    (0.60415_f32, 0.796871_f32),
    (0.655412_f32, 0.755272_f32),
    (-0.916524_f32, -0.399979_f32),
    (0.763757_f32, -0.645503_f32),
    (0.822298_f32, 0.569058_f32),
    (-0.0283442_f32, -0.999598_f32),
    (-0.644058_f32, -0.764977_f32),
    (-0.390833_f32, -0.920462_f32),
    (-0.984217_f32, -0.176967_f32),
    (0.719987_f32, 0.693987_f32),
    (0.999088_f32, -0.042707_f32),
    (0.754487_f32, 0.656315_f32),
    (0.498312_f32, 0.866998_f32),
    (0.999389_f32, 0.0349481_f32),
    (-0.811845_f32, -0.583873_f32),
    (0.532015_f32, -0.846735_f32),
    (-0.463781_f32, 0.88595_f32),
    (-0.819481_f32, -0.573106_f32),
    (0.637789_f32, 0.770211_f32),
    (0.814253_f32, -0.58051_f32),
    (0.180489_f32, -0.983577_f32),
    (0.988691_f32, -0.149968_f32),
    (0.606847_f32, -0.794819_f32),
    (-0.061914_f32, 0.998081_f32),
    (0.627066_f32, 0.778966_f32),
    (-0.544097_f32, -0.839023_f32),
    (-0.859249_f32, -0.511558_f32),
    (-0.91818_f32, 0.396163_f32),
    (-0.165942_f32, 0.986136_f32),
    (0.992322_f32, 0.123685_f32),
    (0.758555_f32, 0.651609_f32),
    (0.985543_f32, -0.169427_f32),
    (-0.154971_f32, -0.987919_f32),
    (0.245336_f32, 0.969438_f32),
    (-0.522038_f32, -0.852922_f32),
    (-0.33027_f32, -0.943886_f32),
    (0.998067_f32, -0.0621421_f32),
    (0.516758_f32, 0.856132_f32),
    (0.843123_f32, -0.537721_f32),
    (0.44306_f32, 0.896492_f32),
    (-0.814913_f32, 0.579584_f32),
    (-0.336464_f32, -0.941696_f32),
    (0.732896_f32, 0.68034_f32),
    (0.201774_f32, -0.979432_f32),
    (0.741954_f32, 0.670451_f32),
    (-0.469083_f32, 0.883154_f32),
    (0.867784_f32, 0.496941_f32),
    (0.494202_f32, -0.869347_f32),
    (0.9367_f32, -0.350134_f32),
    (0.906328_f32, 0.422575_f32),
    (0.764111_f32, 0.645085_f32),
    (0.631052_f32, -0.77574_f32),
    (-0.0498248_f32, -0.998758_f32),
    (0.974691_f32, -0.223555_f32),
    (0.361405_f32, -0.932409_f32),
    (-0.748625_f32, 0.662994_f32),
    (0.811839_f32, -0.583881_f32),
    (0.303931_f32, 0.952694_f32),
    (-0.992668_f32, -0.120876_f32),
    (-0.996672_f32, -0.0815161_f32),
    (-0.324622_f32, 0.945844_f32),
    (-0.0246385_f32, -0.999696_f32),
    (-0.588361_f32, 0.808598_f32),
    (-0.49898_f32, 0.866613_f32),
    (0.924578_f32, -0.380994_f32),
    (-0.755619_f32, 0.655011_f32),
    (0.92214_f32, -0.386857_f32),
    (0.818638_f32, 0.57431_f32),
    (-0.920451_f32, 0.390857_f32),
    (0.0380205_f32, -0.999277_f32),
    (0.446606_f32, -0.894731_f32),
    (0.722557_f32, -0.691312_f32),
    (0.762113_f32, -0.647444_f32),
    (-0.256731_f32, -0.966483_f32),
    (0.471301_f32, -0.881972_f32),
    (-0.530869_f32, -0.847454_f32),
    (-0.749876_f32, 0.661579_f32),
    (-0.593767_f32, -0.804637_f32),
    (-0.834805_f32, 0.550545_f32),
    (0.748843_f32, 0.662747_f32),
    (-0.70794_f32, 0.706273_f32),
    (-0.0503228_f32, -0.998733_f32),
    (0.402884_f32, -0.915251_f32),
    (0.0945791_f32, -0.995517_f32),
    (-0.390889_f32, -0.920438_f32),
    (-0.0994705_f32, -0.995041_f32),
    (-0.767207_f32, -0.641399_f32),
    (-0.563485_f32, -0.826126_f32),
    (-0.521859_f32, 0.853032_f32),
    (0.503637_f32, 0.863915_f32),
    (-0.739851_f32, -0.672771_f32),
    (0.442624_f32, -0.896707_f32),
    (-0.997671_f32, 0.0682121_f32),
    (0.776117_f32, -0.630589_f32),
    (-0.964601_f32, -0.263713_f32),
    (-0.656053_f32, 0.754715_f32),
    (-0.865578_f32, 0.500775_f32),
    (-0.586255_f32, -0.810127_f32),
    (0.0425286_f32, -0.999095_f32),
    (0.656339_f32, 0.754466_f32),
    (-0.341906_f32, 0.939734_f32),
    (-0.605904_f32, 0.795538_f32),
    (-0.658238_f32, -0.752809_f32),
    (-0.652856_f32, 0.757482_f32),
    (-0.994554_f32, 0.104219_f32),
    (0.176725_f32, -0.98426_f32),
    (-0.231945_f32, 0.972729_f32),
    (-0.997717_f32, -0.0675303_f32),
    (-0.997805_f32, 0.0662198_f32),
    (0.661155_f32, 0.75025_f32),
    (0.999669_f32, -0.02573_f32),
    (-0.946982_f32, -0.321287_f32),
    (-0.587897_f32, 0.808936_f32),
    (0.957862_f32, -0.28723_f32),
    (0.613392_f32, -0.789779_f32),
    (-0.956489_f32, 0.291768_f32),
    (0.169829_f32, -0.985474_f32),
    (-0.943551_f32, -0.331226_f32),
    (0.416834_f32, 0.908983_f32),
    (0.684727_f32, -0.7288_f32),
    (0.952329_f32, 0.305074_f32),
    (-0.328392_f32, 0.944541_f32),
    (0.943344_f32, 0.331816_f32),
    (0.650872_f32, 0.759187_f32),
    (-0.59941_f32, -0.800442_f32),
    (-0.768448_f32, -0.639912_f32),
    (0.539894_f32, 0.841733_f32),
    (0.606048_f32, -0.795428_f32),
    (0.403588_f32, 0.914941_f32),
    (-0.838111_f32, 0.545499_f32),
    (0.976157_f32, -0.217066_f32),
    (-0.995495_f32, -0.0948107_f32),
    (-0.943792_f32, 0.330539_f32),
    (-0.990415_f32, 0.138123_f32),
    (0.281355_f32, 0.959604_f32),
    (0.371208_f32, 0.92855_f32),
    (-0.4711_f32, 0.88208_f32),
    (-0.999459_f32, -0.0328787_f32),
    (-0.988179_f32, 0.153305_f32),
    (-0.843124_f32, -0.537718_f32),
    (0.997108_f32, 0.0759971_f32),
    (-0.268201_f32, -0.963363_f32),
    (0.0457651_f32, 0.998952_f32),
    (-0.983762_f32, 0.179479_f32),
    (-0.439728_f32, -0.898131_f32),
    (0.162945_f32, -0.986635_f32),
    (-0.055868_f32, 0.998438_f32),
    (-0.384381_f32, 0.923174_f32),
    (0.744138_f32, -0.668026_f32),
    (-0.0706482_f32, -0.997501_f32),
    (0.831219_f32, 0.555945_f32),
    (0.711624_f32, 0.70256_f32),
    (0.161772_f32, -0.986828_f32),
    (0.8387_f32, 0.544593_f32),
    (-0.418108_f32, 0.908397_f32),
    (0.412208_f32, 0.91109_f32),
    (-0.986564_f32, 0.163375_f32),
    (0.925222_f32, 0.379427_f32),
    (0.411092_f32, -0.911594_f32),
    (-0.103885_f32, -0.994589_f32),
    (-0.923358_f32, -0.383941_f32),
    (-0.761339_f32, 0.648355_f32),
    (-0.887774_f32, -0.46028_f32),
    (0.755699_f32, -0.654919_f32),
    (0.597832_f32, -0.801622_f32),
    (0.542946_f32, -0.839768_f32),
    (0.999842_f32, -0.0177525_f32),
    (-0.954864_f32, 0.297044_f32),
    (-0.999961_f32, -0.00886517_f32),
    (0.929688_f32, -0.368347_f32),
    (0.699205_f32, 0.714921_f32),
    (-0.894118_f32, 0.447831_f32),
    (-0.903496_f32, 0.428597_f32),
    (0.362774_f32, -0.931877_f32),
    (0.850158_f32, -0.526527_f32),
    (0.523993_f32, -0.851722_f32),
    (-0.735767_f32, 0.677235_f32),
    (-0.999673_f32, -0.02558_f32),
    (-0.954685_f32, 0.297619_f32),
    (-0.195193_f32, -0.980765_f32),
    (0.0672577_f32, -0.997736_f32),
    (-0.659842_f32, -0.751405_f32),
    (0.779366_f32, 0.626569_f32),
    (0.751561_f32, -0.659663_f32),
    (-0.34458_f32, -0.938757_f32),
    (0.316872_f32, 0.948468_f32),
    (0.953601_f32, 0.301074_f32),
    (0.363243_f32, 0.931694_f32),
    (-0.537299_f32, -0.843392_f32),
    (0.996839_f32, 0.0794518_f32),
    (0.950462_f32, -0.310842_f32),
    (0.224973_f32, 0.974365_f32),
    (-0.772894_f32, -0.634535_f32),
    (-0.94373_f32, -0.330718_f32),
    (0.930603_f32, 0.36603_f32),
    (0.994059_f32, 0.10884_f32),
    (0.845518_f32, -0.533947_f32),
    (0.988122_f32, -0.15367_f32),
    (0.880641_f32, -0.473785_f32),
    (0.783488_f32, -0.621407_f32),
    (0.72854_f32, 0.685004_f32),
    (-0.772294_f32, 0.635265_f32),
    (0.662444_f32, 0.749111_f32),
    (-0.0649291_f32, 0.99789_f32),
    (-0.285125_f32, 0.95849_f32),
    (-0.673637_f32, -0.739062_f32),
    (-0.394791_f32, -0.918771_f32),
    (-0.938677_f32, -0.344798_f32),
    (-0.708925_f32, -0.705284_f32),
    (0.31416_f32, 0.94937_f32),
    (-0.113645_f32, 0.993521_f32),
    (-0.296446_f32, 0.95505_f32),
    (0.670712_f32, -0.741718_f32),
    (-0.605825_f32, 0.795598_f32),
    (0.996229_f32, 0.0867643_f32),
    (0.686613_f32, -0.727023_f32),
    (-0.740136_f32, 0.672457_f32),
    (0.876977_f32, 0.480532_f32),
    (-0.561046_f32, 0.827785_f32),
    (0.414562_f32, -0.910021_f32),
    (-0.645953_f32, -0.763377_f32),
    (0.802263_f32, 0.596971_f32),
    (-0.854981_f32, 0.51866_f32),
    (-0.769916_f32, 0.638145_f32),
    (0.648047_f32, 0.7616_f32),
    (-0.773406_f32, 0.633911_f32),
    (-0.252579_f32, -0.967576_f32),
    (0.962561_f32, -0.271064_f32),
    (0.959193_f32, -0.282751_f32),
    (0.727508_f32, 0.686099_f32),
    (-0.667916_f32, 0.744236_f32),
    (-0.599333_f32, 0.8005_f32),
    (0.622504_f32, 0.782616_f32),
    (0.375433_f32, -0.92685_f32),
    (-0.9972_f32, 0.0747742_f32),
    (-0.879355_f32, 0.476166_f32),
    (0.409574_f32, 0.912277_f32),
    (0.747405_f32, -0.664369_f32),
    (-0.940177_f32, -0.340687_f32),
    (0.562532_f32, 0.826776_f32),
    (-0.929015_f32, 0.370041_f32),
    (0.0978642_f32, 0.9952_f32),
    (0.916896_f32, -0.399125_f32),
    (-0.608038_f32, -0.793908_f32),
    (-0.845653_f32, -0.533732_f32),
    (-0.455945_f32, -0.890008_f32),
    (0.923484_f32, -0.383637_f32),
    (0.354901_f32, 0.934904_f32),
    (0.319134_f32, -0.94771_f32),
    (0.769603_f32, -0.638523_f32),
    (-0.899207_f32, 0.437524_f32),
    (0.666669_f32, -0.745354_f32),
    (0.142655_f32, -0.989772_f32),
    (-0.892_f32, -0.452035_f32),
    (-0.999915_f32, -0.0130018_f32),
    (-0.87033_f32, 0.492469_f32),
    (0.156511_f32, 0.987676_f32),
    (-0.146752_f32, 0.989173_f32),
    (-0.809057_f32, 0.58773_f32),
    (-0.497325_f32, -0.867565_f32),
    (-0.258455_f32, -0.966023_f32),
    (-0.863292_f32, -0.504705_f32),
    (-0.976343_f32, -0.216225_f32),
    (-0.257626_f32, -0.966245_f32),
    (0.809568_f32, -0.587027_f32),
    (0.582491_f32, 0.812837_f32),
    (-0.997088_f32, -0.0762565_f32),
    (-0.878262_f32, 0.47818_f32),
    (-0.165343_f32, -0.986236_f32),
    (0.0455161_f32, 0.998964_f32),
    (-0.570664_f32, 0.821184_f32),
    (0.658564_f32, 0.752525_f32),
    (0.319839_f32, -0.947472_f32),
    (-0.643905_f32, -0.765106_f32),
    (-0.590256_f32, 0.807216_f32),
    (0.512137_f32, 0.858904_f32),
    (-0.998558_f32, 0.0536785_f32),
    (0.373964_f32, -0.927443_f32),
    (0.633108_f32, -0.774063_f32),
    (-0.968108_f32, 0.250531_f32),
    (0.787337_f32, 0.616523_f32),
    (0.698247_f32, -0.715857_f32),
    (0.98937_f32, 0.145419_f32),
    (0.582241_f32, -0.813016_f32),
    (0.359617_f32, 0.9331_f32),
    (-0.758164_f32, -0.652064_f32),
    (0.635102_f32, 0.772429_f32),
    (-0.0254028_f32, -0.999677_f32),
    (0.266382_f32, -0.963868_f32),
    (0.660974_f32, -0.750409_f32),
    (0.585176_f32, -0.810906_f32),
    (-0.98243_f32, 0.186631_f32),
    (0.777252_f32, -0.629189_f32),
    (-0.0267382_f32, 0.999642_f32),
    (-0.95591_f32, 0.293661_f32),
    (0.70368_f32, -0.710517_f32),
    (0.732467_f32, -0.680803_f32),
    (0.854099_f32, -0.52011_f32),
    (0.536151_f32, 0.844122_f32),
    (-0.00781503_f32, 0.999969_f32),
    (-0.534447_f32, 0.845202_f32),
    (0.297782_f32, 0.954634_f32),
    (0.905724_f32, 0.423868_f32),
    (0.115617_f32, -0.993294_f32),
    (-0.993408_f32, -0.114636_f32),
    (0.156977_f32, 0.987602_f32),
    (0.639408_f32, -0.768868_f32),
    (-0.995898_f32, 0.0904832_f32),
    (-0.956372_f32, 0.29215_f32),
    (0.990545_f32, 0.137189_f32),
    (0.659118_f32, 0.75204_f32),
    (-0.0398563_f32, -0.999205_f32),
    (-0.679682_f32, -0.733507_f32),
    (-0.540035_f32, 0.841643_f32),
    (-0.0501135_f32, -0.998744_f32),
    (-0.196305_f32, -0.980543_f32),
    (0.56964_f32, 0.821894_f32),
    (-0.703653_f32, 0.710544_f32),
    (0.162676_f32, -0.98668_f32),
    (-0.919545_f32, 0.392984_f32),
    (0.805179_f32, 0.593031_f32),
    (0.998757_f32, 0.0498406_f32),
    (0.358168_f32, 0.933657_f32),
    (-0.611152_f32, -0.791513_f32),
    (-0.440479_f32, 0.897763_f32),
    (0.292587_f32, 0.956239_f32),
    (-0.217415_f32, -0.976079_f32),
    (-0.252622_f32, -0.967565_f32),
    (-0.679998_f32, -0.733214_f32),
    (0.402652_f32, -0.915353_f32),
    (-0.993189_f32, 0.116516_f32),
    (0.0634956_f32, 0.997982_f32),
    (0.432309_f32, -0.901725_f32),
    (0.923434_f32, 0.383756_f32),
    (-0.502058_f32, 0.864834_f32),
    (0.935584_f32, -0.353105_f32),
    (0.912111_f32, -0.409944_f32),
    (-0.298643_f32, 0.954365_f32),
    (-0.796165_f32, 0.60508_f32),
    (-0.741295_f32, -0.671179_f32),
    (0.856386_f32, 0.516336_f32),
    (-0.515876_f32, -0.856663_f32),
    (0.994745_f32, -0.102384_f32),
    (0.648698_f32, -0.761046_f32),
    (-0.999675_f32, -0.0254761_f32),
    (-0.130115_f32, 0.991499_f32),
    (-0.998787_f32, 0.0492413_f32),
    (0.27449_f32, -0.96159_f32),
    (-0.996501_f32, -0.0835749_f32),
    (-0.387182_f32, 0.922003_f32),
    (0.701006_f32, -0.713156_f32),
    (0.98733_f32, -0.158679_f32),
    (-0.713847_f32, 0.700302_f32),
    (-0.329606_f32, 0.944118_f32),
    (0.279362_f32, 0.960186_f32),
    (-0.968574_f32, -0.248726_f32),
    (-0.68131_f32, 0.731995_f32),
    (0.220789_f32, -0.975322_f32),
    (-0.985566_f32, -0.169292_f32),
    (0.0132834_f32, 0.999912_f32),
    (-0.422317_f32, 0.906448_f32),
    (-0.772023_f32, -0.635594_f32),
    (0.842036_f32, -0.539421_f32),
    (-0.803312_f32, 0.595558_f32),
    (0.725035_f32, 0.688712_f32),
    (0.328206_f32, 0.944606_f32),
    (0.711898_f32, 0.702283_f32),
    (-0.691674_f32, 0.72221_f32),
    (-0.871274_f32, 0.490797_f32),
    (0.213736_f32, 0.976891_f32),
    (0.255845_f32, 0.966718_f32),
    (0.883381_f32, 0.468656_f32),
    (-0.596736_f32, 0.802437_f32),
    (0.779861_f32, 0.625953_f32),
    (-0.607233_f32, 0.794524_f32),
    (-0.944679_f32, -0.327996_f32),
    (0.851219_f32, -0.52481_f32),
    (-0.859337_f32, -0.51141_f32),
    (0.953486_f32, -0.301437_f32),
    (0.512244_f32, -0.85884_f32),
    (0.160393_f32, 0.987053_f32),
    (0.752002_f32, 0.659161_f32),
    (0.999882_f32, 0.0153624_f32),
    (0.778011_f32, -0.628251_f32),
    (0.9293_f32, -0.369326_f32),
    (-0.605896_f32, 0.795544_f32),
    (0.633164_f32, 0.774017_f32),
    (-0.923382_f32, -0.383883_f32),
    (-0.790911_f32, 0.611931_f32),
    (0.673492_f32, 0.739195_f32),
    (-0.784902_f32, 0.61962_f32),
    (0.289472_f32, 0.957187_f32),
    (0.605387_f32, -0.795931_f32),
    (-0.459844_f32, -0.888_f32),
    (-0.990035_f32, 0.140822_f32),
    (-0.686367_f32, -0.727256_f32),
    (-0.549857_f32, 0.835259_f32),
    (0.90982_f32, 0.415003_f32),
    (-0.42105_f32, 0.907037_f32),
    (-0.07295_f32, 0.997336_f32),
    (-0.24021_f32, 0.970721_f32),
    (0.993154_f32, -0.116813_f32),
    (-0.59563_f32, -0.803259_f32),
    (0.526545_f32, -0.850147_f32),
    (0.9987_f32, -0.0509667_f32),
    (-0.85017_f32, -0.526508_f32),
    (-0.818838_f32, -0.574025_f32),
    (0.982094_f32, 0.188394_f32),
    (0.577634_f32, -0.816296_f32),
    (-0.418394_f32, -0.908265_f32),
    (0.62868_f32, -0.777664_f32),
    (-0.118173_f32, -0.992993_f32),
    (0.896113_f32, 0.443827_f32),
    (-0.159857_f32, -0.98714_f32),
    (0.750036_f32, 0.661397_f32),
    (0.745659_f32, 0.666328_f32),
    (-0.938865_f32, -0.344285_f32),
    (-0.583143_f32, 0.812369_f32),
    (0.479122_f32, 0.877748_f32),
    (-0.869898_f32, -0.493231_f32),
    (-0.791797_f32, 0.610784_f32),
    (0.35785_f32, 0.933779_f32),
    (0.25248_f32, -0.967602_f32),
    (-0.573942_f32, -0.818896_f32),
    (-0.930426_f32, -0.366479_f32),
    (-0.378158_f32, 0.925741_f32),
    (-0.942114_f32, 0.335293_f32),
    (0.647836_f32, -0.76178_f32),
    (0.814174_f32, 0.580622_f32),
    (0.023769_f32, 0.999717_f32),
    (0.112026_f32, -0.993705_f32),
    (0.659378_f32, -0.751811_f32),
    (-0.615064_f32, -0.788477_f32),
    (-0.00328067_f32, 0.999995_f32),
    (0.902263_f32, -0.431186_f32),
    (0.201174_f32, -0.979556_f32),
    (0.541589_f32, 0.840644_f32),
    (-0.996013_f32, 0.0892081_f32),
    (0.987237_f32, 0.159261_f32),
    (-0.692353_f32, 0.721559_f32),
    (0.940855_f32, -0.338809_f32),
    (0.164224_f32, -0.986423_f32),
    (0.0618662_f32, 0.998084_f32),
    (0.784694_f32, 0.619883_f32),
    (0.156281_f32, -0.987713_f32),
    (-0.424548_f32, -0.905405_f32),
    (0.927622_f32, 0.37352_f32),
    (-0.923711_f32, 0.383089_f32),
    (0.708767_f32, -0.705442_f32),
    (-0.941076_f32, 0.338195_f32),
    (-0.268226_f32, 0.963356_f32),
    (0.653964_f32, 0.756526_f32),
    (0.983767_f32, -0.179452_f32),
    (-0.480029_f32, 0.877253_f32),
    (-0.845565_f32, -0.533873_f32),
    (-0.768586_f32, -0.639746_f32),
    (0.208936_f32, 0.977929_f32),
    (0.512539_f32, -0.858664_f32),
    (0.988163_f32, -0.153409_f32),
    (0.780816_f32, 0.624761_f32),
    (-0.232748_f32, 0.972537_f32),
    (0.988528_f32, -0.151035_f32),
    (-0.106602_f32, -0.994302_f32),
    (-0.633295_f32, 0.773911_f32),
    (0.322068_f32, -0.946717_f32),
    (0.989632_f32, -0.143627_f32),
    (0.973492_f32, -0.228721_f32),
    (0.998266_f32, 0.0588631_f32),
    (-0.145619_f32, 0.989341_f32),
]; // ASPX_NOISE_TABLE

/// Output of the A-SPX noise generator — one `qmf_noise[sb][ts]`
/// complex matrix plus the stateful `noise_idx_prev` used to seed
/// the next A-SPX interval's `noise_idx()` helper (Pseudocode 103).
///
/// `qmf_noise[sb]` is indexed by the **absolute** QMF subband number
/// (0..NUM_QMF_SUBBANDS). Entries outside the A-SPX range `sbx..sbz`
/// are left zero so the assembler can add directly into the high-band
/// matrix without indexing gymnastics.
#[derive(Debug, Clone, Default)]
pub struct QmfNoise {
    /// `qmf_noise[sb][ts] = noise_lev_sb_adj[sb-sbx][atsg] *
    /// NoiseTable[noise_idx(sb, ts)]` per Pseudocode 102. Stored with
    /// absolute subband indexing (0..num_qmf_subbands).
    pub qmf_noise: Vec<Vec<(f32, f32)>>,
    /// Last `noise_idx` produced during the generator run, keyed by
    /// absolute subband. Future rounds will feed this back in as
    /// `noise_idx_prev` so successive A-SPX intervals don't repeat.
    pub last_indices: Vec<Vec<u32>>,
}

/// State carried across A-SPX intervals for the noise generator.
#[derive(Debug, Clone, Default)]
pub struct NoiseGenState {
    /// `noise_idx_prev[sb][ts]` — per Pseudocode 103, the last
    /// `noise_idx` from the previous A-SPX interval. `None` on the
    /// first frame ("master_reset" in Pseudocode 103 == 1).
    pub prev: Option<Vec<Vec<u32>>>,
}

impl NoiseGenState {
    /// Construct a fresh state (first-frame, `master_reset` semantics).
    pub fn new() -> Self {
        Self { prev: None }
    }

    /// Reset the state to the "master_reset == 1" initial condition
    /// (§5.7.6.3.1.1).
    pub fn reset(&mut self) {
        self.prev = None;
    }
}

/// Compute `noise_idx(sb, ts)` per ETSI TS 103 190-1 §5.7.6.4.3
/// Pseudocode 103.
///
/// * `sb` — absolute QMF subband index (0..num_qmf_subbands).
/// * `ts` — absolute QMF time-slot index.
/// * `atsg_sig0` — `atsg_sig[0]` (start of the current A-SPX
///   interval in A-SPX time slots).
/// * `num_ts_in_ats` — QMF slots per A-SPX slot (1 or 2,
///   Table 192).
/// * `num_sb_aspx` — `sbz - sbx`.
/// * `noise_idx_prev` — previous A-SPX interval's `noise_idx`
///   output for (sb, ts), or `None` if this is the first frame
///   (master_reset == 1).
#[inline]
pub fn noise_idx(
    sb_absolute: u32,
    ts: u32,
    atsg_sig0: u32,
    num_ts_in_ats: u32,
    num_sb_aspx: u32,
    noise_idx_prev: Option<u32>,
) -> u32 {
    let mut idx: u32 = noise_idx_prev.unwrap_or(0);
    // ts - atsg_sig[0] * num_ts_in_ats: per Pseudocode 103.
    // Spec reads "indexNoise += num_sb_aspx * (ts - atsg_sig[0])"
    // in A-SPX-timeslot form when used with `ts` the outer QMF
    // timeslot — the outer loop starts at atsg_sig[0]*num_ts_in_ats,
    // so the delta here is expressed in QMF slots.
    let ts_off = ts.saturating_sub(atsg_sig0.saturating_mul(num_ts_in_ats));
    idx = idx.wrapping_add(num_sb_aspx.wrapping_mul(ts_off));
    idx = idx.wrapping_add(sb_absolute + 1);
    idx % (NOISE_TABLE_LEN as u32)
}

/// Generate the complex noise QMF matrix per ETSI TS 103 190-1
/// §5.7.6.4.3 Pseudocode 102.
///
/// * `noise_lev_sb_adj[sb][atsg_noise]` — per-subband, per-noise-
///   envelope boosted noise level (shape `num_sb_aspx × num_atsg_noise`).
///   This is the output of the envelope-adjustment tool's Pseudocode 94
///   plus Pseudocode 101 boost.
/// * `atsg_sig` — signal-envelope borders (A-SPX timeslots,
///   `num_atsg_sig + 1` entries).
/// * `atsg_noise` — noise-envelope borders (A-SPX timeslots,
///   `num_atsg_noise + 1` entries).
/// * `num_ts_in_ats` — 1 or 2 (Table 192).
/// * `num_qmf_subbands` — almost always 64 in AC-4.
/// * `sbx` — crossover subband.
/// * `num_sb_aspx` — `sbz - sbx`.
/// * `state` — cross-interval noise index state (advance on return).
///
/// The atsg index walks both `atsg_sig` (outer loop) and `atsg_noise`
/// (the noise matrix has a coarser time axis); the matching
/// `atsg_noise` column for a given `atsg_sig` value is looked up by
/// finding the smallest `atsg_noise[i+1]` that exceeds the current
/// signal border — this mirrors Pseudocode 102 where the outer loop
/// variable is `atsg` (signal) but the indexing into
/// `noise_lev_sb_adj` uses whichever noise envelope covers the slot.
#[allow(clippy::too_many_arguments)]
pub fn generate_qmf_noise(
    noise_lev_sb_adj: &[Vec<f32>],
    atsg_sig: &[u32],
    atsg_noise: &[u32],
    num_ts_in_ats: u32,
    num_qmf_subbands: u32,
    sbx: u32,
    num_sb_aspx: u32,
    state: &mut NoiseGenState,
) -> QmfNoise {
    let num_atsg_sig = atsg_sig.len().saturating_sub(1);
    if num_atsg_sig == 0 || num_sb_aspx == 0 {
        return QmfNoise::default();
    }
    let ts_start = atsg_sig[0].saturating_mul(num_ts_in_ats);
    let ts_end = atsg_sig[num_atsg_sig].saturating_mul(num_ts_in_ats);
    // Allocate the full QMF matrix width (absolute subband indexing).
    let ts_total = ts_end as usize;
    let mut qmf_noise: Vec<Vec<(f32, f32)>> = (0..num_qmf_subbands)
        .map(|_| vec![(0.0_f32, 0.0_f32); ts_total])
        .collect();
    let mut last_indices: Vec<Vec<u32>> = (0..num_qmf_subbands)
        .map(|_| vec![0_u32; ts_total])
        .collect();
    // Walk the full interval once. The signal envelope index `atsg`
    // increments whenever ts crosses the next atsg_sig border.
    let mut atsg: usize = 0;
    // Current noise-envelope index, advanced alongside atsg_sig when
    // the signal envelope crosses the next atsg_noise border.
    let mut atsg_noise_idx: usize = 0;
    // Precompute ts boundaries in QMF slots for both sig and noise.
    for ts in ts_start..ts_end {
        // Advance signal atsg if ts crossed the next border.
        // atsg_sig[atsg+1] * num_ts_in_ats == ts marks the transition.
        while atsg + 1 < num_atsg_sig && ts >= atsg_sig[atsg + 1].saturating_mul(num_ts_in_ats) {
            atsg += 1;
        }
        // Advance noise atsg similarly.
        let num_atsg_noise = atsg_noise.len().saturating_sub(1);
        while atsg_noise_idx + 1 < num_atsg_noise
            && ts >= atsg_noise[atsg_noise_idx + 1].saturating_mul(num_ts_in_ats)
        {
            atsg_noise_idx += 1;
        }
        for sb in 0..(num_sb_aspx as usize) {
            let sb_abs = sb + sbx as usize;
            if sb_abs >= qmf_noise.len() {
                break;
            }
            // Pull the previous index for (sb_abs, ts) if available.
            let prev = state.prev.as_ref().and_then(|mat| {
                mat.get(sb_abs)
                    .and_then(|row| row.get(ts as usize).copied())
            });
            let idx = noise_idx(
                sb_abs as u32,
                ts,
                atsg_sig[0],
                num_ts_in_ats,
                num_sb_aspx,
                prev,
            );
            last_indices[sb_abs][ts as usize] = idx;
            let (nre, nim) = ASPX_NOISE_TABLE[idx as usize];
            let lev = noise_lev_sb_adj
                .get(sb)
                .and_then(|row| row.get(atsg_noise_idx))
                .copied()
                .unwrap_or(0.0);
            qmf_noise[sb_abs][ts as usize] = (lev * nre, lev * nim);
        }
    }
    // Advance state: store last_indices for the next call.
    state.prev = Some(last_indices.clone());
    QmfNoise {
        qmf_noise,
        last_indices,
    }
}

/// Add the generated `qmf_noise` into an existing QMF high-band
/// matrix `y` in place per ETSI TS 103 190-1 §5.7.6.4.5 Pseudocode 107.
///
/// `y[sb][ts]` and `qmf_noise[sb][ts]` must share the same layout
/// (absolute subband indexing, same time-slot count). The addition
/// runs only over `sbx..sbz` and `atsg_sig[0]*num_ts_in_ats
/// ..atsg_sig[num_atsg_sig]*num_ts_in_ats`.
pub fn add_qmf_noise(
    y: &mut [Vec<(f32, f32)>],
    qmf_noise: &QmfNoise,
    atsg_sig: &[u32],
    num_ts_in_ats: u32,
    sbx: u32,
    sbz: u32,
) {
    let num_atsg_sig = atsg_sig.len().saturating_sub(1);
    if num_atsg_sig == 0 {
        return;
    }
    let ts_start = atsg_sig[0].saturating_mul(num_ts_in_ats) as usize;
    let ts_end = atsg_sig[num_atsg_sig].saturating_mul(num_ts_in_ats) as usize;
    for sb in (sbx as usize)..(sbz as usize) {
        if sb >= y.len() || sb >= qmf_noise.qmf_noise.len() {
            break;
        }
        let src = &qmf_noise.qmf_noise[sb];
        let dst = &mut y[sb];
        let hi = ts_end.min(dst.len()).min(src.len());
        for ts in ts_start..hi {
            dst[ts].0 += src[ts].0;
            dst[ts].1 += src[ts].1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_table_has_512_entries() {
        assert_eq!(ASPX_NOISE_TABLE.len(), 512);
        assert_eq!(NOISE_TABLE_LEN, 512);
    }

    #[test]
    fn noise_table_average_energy_is_unity() {
        // Annex D.2 says "average energy of 1". Verify to three decimals.
        let sum: f64 = ASPX_NOISE_TABLE
            .iter()
            .map(|&(re, im)| (re as f64).powi(2) + (im as f64).powi(2))
            .sum();
        let avg = sum / (ASPX_NOISE_TABLE.len() as f64);
        assert!((avg - 1.0).abs() < 0.01, "avg energy = {avg}");
    }

    #[test]
    fn noise_idx_first_frame_follows_pseudocode_103() {
        // master_reset branch: indexNoise = 0, then
        // indexNoise += num_sb_aspx * (ts - atsg_sig[0])
        // indexNoise += sb + 1
        // returns % 512.
        //
        // With atsg_sig0=0, num_ts_in_ats=1, num_sb_aspx=8:
        //   noise_idx(sb=0, ts=0) = 0 + 0 + 1 = 1
        //   noise_idx(sb=3, ts=0) = 0 + 0 + 4 = 4
        //   noise_idx(sb=0, ts=1) = 0 + 8 + 1 = 9
        //   noise_idx(sb=7, ts=2) = 0 + 16 + 8 = 24
        assert_eq!(noise_idx(0, 0, 0, 1, 8, None), 1);
        assert_eq!(noise_idx(3, 0, 0, 1, 8, None), 4);
        assert_eq!(noise_idx(0, 1, 0, 1, 8, None), 9);
        assert_eq!(noise_idx(7, 2, 0, 1, 8, None), 24);
    }

    #[test]
    fn noise_idx_modulo_wraps() {
        // Make sure the 512-modulus wraps cleanly.
        // With num_sb_aspx=100, ts=10, atsg_sig0=0, num_ts_in_ats=1:
        //   idx = 0 + 100*10 + 50 + 1 = 1051 % 512 = 27.
        assert_eq!(noise_idx(50, 10, 0, 1, 100, None), 27);
    }

    #[test]
    fn noise_idx_uses_previous_state() {
        // previous index becomes the seed for the current call.
        // idx = prev + num_sb_aspx*(ts-atsg_sig0*num_ts_in_ats) + sb+1
        // idx = (100 + 8*4 + 4) mod 512 = 136.
        assert_eq!(noise_idx(3, 4, 0, 1, 8, Some(100)), 136);
    }

    #[test]
    fn generator_produces_nonzero_noise_in_aspx_range() {
        // num_sb_aspx = 4, num_ts_in_ats = 1, sbx = 8, sbz = 12.
        // One signal + one noise envelope spanning 4 atsg time slots.
        let atsg_sig = vec![0_u32, 4];
        let atsg_noise = vec![0_u32, 4];
        let noise_lev_sb_adj = vec![vec![1.0_f32], vec![1.0_f32], vec![1.0_f32], vec![1.0_f32]];
        let mut state = NoiseGenState::new();
        let out = generate_qmf_noise(
            &noise_lev_sb_adj,
            &atsg_sig,
            &atsg_noise,
            1,
            64,
            8,
            4,
            &mut state,
        );
        // Shape: 64 rows × 4 slots. Only sb in 8..12 should be non-zero.
        assert_eq!(out.qmf_noise.len(), 64);
        assert_eq!(out.qmf_noise[0].len(), 4);
        for sb in 0..8 {
            for ts in 0..4 {
                assert_eq!(out.qmf_noise[sb][ts], (0.0, 0.0));
            }
        }
        let mut any_nz = false;
        for sb in 8..12 {
            for ts in 0..4 {
                let (re, im) = out.qmf_noise[sb][ts];
                if re.abs() > 1e-6 || im.abs() > 1e-6 {
                    any_nz = true;
                }
                // The magnitudes should match |NOISE_TABLE[idx]| (level=1.0).
                let idx = noise_idx(sb as u32, ts as u32, 0, 1, 4, None);
                let (exp_re, exp_im) = ASPX_NOISE_TABLE[idx as usize];
                assert!((re - exp_re).abs() < 1e-5);
                assert!((im - exp_im).abs() < 1e-5);
            }
        }
        assert!(any_nz);
        for sb in 12..64 {
            for ts in 0..4 {
                assert_eq!(out.qmf_noise[sb][ts], (0.0, 0.0));
            }
        }
        assert!(state.prev.is_some());
    }

    #[test]
    fn generator_level_scales_noise_samples() {
        // Gain of 2.0 should double the table values (modulo floating error).
        let atsg_sig = vec![0_u32, 2];
        let atsg_noise = vec![0_u32, 2];
        let noise_lev_sb_adj = vec![vec![2.0_f32]; 2];
        let mut state = NoiseGenState::new();
        let out = generate_qmf_noise(
            &noise_lev_sb_adj,
            &atsg_sig,
            &atsg_noise,
            1,
            64,
            4,
            2,
            &mut state,
        );
        for sb in 4..6 {
            for ts in 0..2 {
                let (re, im) = out.qmf_noise[sb][ts];
                let idx = noise_idx(sb as u32, ts as u32, 0, 1, 2, None);
                let (exp_re, exp_im) = ASPX_NOISE_TABLE[idx as usize];
                assert!((re - 2.0 * exp_re).abs() < 1e-5);
                assert!((im - 2.0 * exp_im).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn generator_picks_correct_noise_envelope() {
        // Two noise envelopes covering different time ranges with
        // different levels — confirm the atsg_noise walker picks the
        // right column.
        let atsg_sig = vec![0_u32, 4];
        // noise envelope 0 covers atsg 0..2 (ts 0..2), envelope 1
        // covers ts 2..4.
        let atsg_noise = vec![0_u32, 2, 4];
        let noise_lev_sb_adj = vec![vec![1.0_f32, 3.0_f32], vec![1.0_f32, 3.0_f32]];
        let mut state = NoiseGenState::new();
        let out = generate_qmf_noise(
            &noise_lev_sb_adj,
            &atsg_sig,
            &atsg_noise,
            1,
            64,
            4,
            2,
            &mut state,
        );
        // First two slots should use level 1.0, last two should use 3.0.
        for ts in 0..2 {
            for sb in 4..6 {
                let (re, _im) = out.qmf_noise[sb][ts];
                let idx = noise_idx(sb as u32, ts as u32, 0, 1, 2, None);
                let (exp_re, _) = ASPX_NOISE_TABLE[idx as usize];
                assert!((re - 1.0 * exp_re).abs() < 1e-5);
            }
        }
        for ts in 2..4 {
            for sb in 4..6 {
                let (re, _im) = out.qmf_noise[sb][ts];
                let idx = noise_idx(sb as u32, ts as u32, 0, 1, 2, None);
                let (exp_re, _) = ASPX_NOISE_TABLE[idx as usize];
                assert!((re - 3.0 * exp_re).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn add_qmf_noise_respects_range() {
        let atsg_sig = vec![0_u32, 2];
        let atsg_noise = vec![0_u32, 2];
        let noise_lev_sb_adj = vec![vec![1.0_f32]; 2];
        let mut state = NoiseGenState::new();
        let out = generate_qmf_noise(
            &noise_lev_sb_adj,
            &atsg_sig,
            &atsg_noise,
            1,
            64,
            4,
            2,
            &mut state,
        );
        let mut y: Vec<Vec<(f32, f32)>> = (0..64).map(|_| vec![(0.5_f32, -0.5_f32); 2]).collect();
        add_qmf_noise(&mut y, &out, &atsg_sig, 1, 4, 6);
        // Subbands 0..4 and 6..64 unchanged (still 0.5, -0.5).
        for sb in 0..4 {
            for ts in 0..2 {
                assert_eq!(y[sb][ts], (0.5, -0.5));
            }
        }
        for sb in 6..64 {
            for ts in 0..2 {
                assert_eq!(y[sb][ts], (0.5, -0.5));
            }
        }
        // Subbands 4..6 have noise added on top of (0.5, -0.5).
        for sb in 4..6 {
            for ts in 0..2 {
                let idx = noise_idx(sb as u32, ts as u32, 0, 1, 2, None);
                let (exp_re, exp_im) = ASPX_NOISE_TABLE[idx as usize];
                assert!((y[sb][ts].0 - (0.5 + exp_re)).abs() < 1e-5);
                assert!((y[sb][ts].1 - (-0.5 + exp_im)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn noise_gen_state_resets_cleanly() {
        let mut state = NoiseGenState::new();
        let atsg_sig = vec![0_u32, 1];
        let atsg_noise = vec![0_u32, 1];
        let noise_lev_sb_adj = vec![vec![1.0_f32]; 2];
        let _ = generate_qmf_noise(
            &noise_lev_sb_adj,
            &atsg_sig,
            &atsg_noise,
            1,
            64,
            2,
            2,
            &mut state,
        );
        assert!(state.prev.is_some());
        state.reset();
        assert!(state.prev.is_none());
    }
}
