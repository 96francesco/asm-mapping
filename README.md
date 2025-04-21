# Description
This project's aim was to develop deep learning models to map artisanal and small-scale mining (ASM) sites using high-resolution satellite imagery and optical-SAR. The models were based on the U-Net architecture and were trained on a dataset of labeled images of ASM sites and non-ASM sites (binary ground truth).

This project stemmed out from my MSc thesis in Geo-Information Science at Wageningen University & Research ([GRS lab](https://www.wur.nl/en/Research-Results/Chair-groups/Environmental-Sciences/Laboratory-of-Geo-information-Science-and-Remote-Sensing.htm)). The thesis was supervised by dr. [Robert Masolele](https://www.wur.nl/en/persons/robert-masolele.htm) and dr. [Johannes Reiche](https://www.wur.nl/nl/en/personen/johannes-reiche.htm).

# Installation 

### Prerequisites

- Python 3.10+
- Poetry (dependency management tool used in this project)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asm-mapping.git
   cd asm-mapping

2. Install dependencies with poetry:
      ```bash
      poetry install
      ```
      or, depending on the Poetry version you installed, run the command below and then paste in the command line its output:
      ```bash
      poetry source env activate
      source /home/your/username/asm-mapping/.venv/bin/activate

3. Activate the virtual environment (not necessary if you used the 'source' command above):
   ```bash
   poetry shell
   ```
