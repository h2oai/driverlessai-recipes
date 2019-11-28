# EXECUTABLES

Malware detection algorithms need to extract features from executable files (PE, ELF, MachO, OAT, DEX, VDEX, and ART formats). The LIEF package (https://github.com/lief-project/LIEF) has tools to parse these files and create features that can be used in training.

## PE Features

Features for PE files are based on EMBER (https://arxiv.org/abs/1804.04637) (https://github.com/endgameinc/ember).

### PEGeneralFeatures

#### ➡️ Code
- [pe_general_features.py](pe_general_features.py)

#### ➡️ Description
Extracts general features from PE files such as size, import/export counts, and other basic features.

#### ➡️ Inputs
- Single text column which contains full paths to PE files on the same machine running DAI

#### ➡️ Outputs
- Multiple numerical columns

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- lief


### PEHeaderFeatures

#### ➡️ Code
- [pe_header_features.py](pe_header_features.py)

#### ➡️ Description
Features derived from the PE file header and option header.

#### ➡️ Inputs
- Single text column which contains full paths to PE files on the same machine running DAI

#### ➡️ Outputs
- 63 numerical columns

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- lief


### PESectionCharacteristics

#### ➡️ Code
- [pe_section_characteristics.py](pe_section_characteristics.py)

#### ➡️ Description
Extracts section characteristics from PE files.

#### ➡️ Inputs
- Single text column which contains full paths to PE files on the same machine running DAI

#### ➡️ Outputs
- Multiple numerical columns

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- lief


### PENormalizedByteCount

#### ➡️ Code
- [pe_normalized_byte_count.py](pe_normalized_byte_count.py)

#### ➡️ Description
The counts for each byte value in a PE file. These counts are then normalized.

#### ➡️ Inputs
- Single text column which contains full paths to PE files on the same machine running DAI

#### ➡️ Outputs
- 256 numerical columns

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- lief


### PEDataDirectoryFeatures

#### ➡️ Code
- [pe_data_directory_features.py](pe_data_directory_features.py)

#### ➡️ Description
Features derived from the PE file data directory

#### ➡️ Inputs
- Single text column which contains full paths to PE files on the same machine running DAI

#### ➡️ Outputs
- 30 numerical columns

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- lief


### PEImportsFeatures

#### ➡️ Code
- [pe_imports_features.py](pe_imports_features.py)

#### ➡️ Description
Features derived from the PE file imports

#### ➡️ Inputs
- Single text column which contains full paths to PE files on the same machine running DAI

#### ➡️ Outputs
- 1280 numerical columns

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- lief


### PEExportsFeatures

#### ➡️ Code
- [pe_exports_features.py](pe_exports_features.py)

#### ➡️ Description
Features derived from the export data section of the PE file.

#### ➡️ Inputs
- Single text column which contains full paths to PE files on the same machine running DAI

#### ➡️ Outputs
- 128 numerical columns

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- lief

