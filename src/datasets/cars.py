import os
import torch
from PIL import Image
from typing import Callable, Optional, Any, Tuple
from torchvision.datasets.vision import VisionDataset
from datasets import load_dataset

class PytorchStanfordCars(VisionDataset):
    """Stanford Cars wrapper feeding directly from public Hugging Face Parquet mirrors"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        hf_split = "train" if split == "train" else "test"
        print(f"Loading Stanford Cars ({split} split) from Hugging Face Parquet Stream...")
        
        # Pulls from an open, unblocked public dataset mirror 
        self.hf_dataset = load_dataset("tanganke/stanford_cars", split=hf_split)
        
        # Hardcoded class mapping matching the official Stanford Cars 196 labels
        self.classes = [
            "AM General Hummer SUV 2000", "Acura RL Sedan 2012", "Acura TL Sedan 2012", "Acura TL Type-S 2008",
            "Acura TSX Sedan 2012", "Acura Integra Type R 2001", "Acura ZDX Hatchback 2012", 
            "Aston Martin V8 Vantage Convertible 2012", "Aston Martin V8 Vantage Coupe 2012", 
            "Aston Martin Virage Convertible 2012", "Aston Martin Virage Coupe 2012", "Audi RS 4 Convertible 2008",
            "Audi A6 Sedan 2011", "Audi TT RS Coupe 2012", "Audi R8 Coupe 2012", "Audi S4 Sedan 2012",
            "Audi S4 Sedan 2007", "Audi S5 Convertible 2012", "Audi S5 Coupe 2012", "Audi S6 Sedan 2011",
            "Audi TT Hatchback 2012", "Audi R8 Spyder 2012", "Audi A4 Sedan 2007", "Audi Q7 SUV 2012",
            "BMW ActiveHybrid 7 Sedan 2012", "BMW 1 Series Convertible 2012", "BMW 1 Series Coupe 2012",
            "BMW 3 Series Sedan 2012", "BMW 3 Series Wagon 2012", "BMW 6 Series Convertible 2012",
            "BMW X5 M SUV 2010", "BMW X6 M SUV 2010", "BMW M3 Coupe 2012", "BMW M5 Sedan 2010",
            "BMW M6 Convertible 2010", "BMW X3 SUV 2012", "BMW Z4 Convertible 2012",
            "Bentley Continental Continental GT Coupe 2007", "Bentley Continental Flying Spur Sedan 2007",
            "Bentley Continental GT Coupe 2012", "Bentley Mulsanne Sedan 2011", "Bentley Arnage Sedan 2009",
            "Bugatti Veyron 16.4 Coupe 2009", "Bugatti Veyron 16.4 Convertible 2009", "Buick Regal GS 2012",
            "Buick Rainier SUV 2007", "Buick Verano Sedan 2012", "Buick Enclave SUV 2012",
            "Cadillac CTS-V Sedan 2012", "Cadillac SRX SUV 2012", "Cadillac Escalade EXT Crew Cab 2007",
            "Chevrolet Silverado 1500 Hybrid Crew Cab 2010", "Chevrolet Silverado 1500 Extended Cab 2012",
            "Chevrolet Silverado 1500 Crew Cab 2012", "Chevrolet Silverado 2500HD Regular Cab 2012",
            "Chevrolet Silverado 3500HD Crew Cab 2012", "Chevrolet Express Cargo Van 2007",
            "Chevrolet Avalanche Crew Cab 2012", "Chevrolet Cobalt SS 2010", "Chevrolet Malibu Sedan 2007",
            "Chevrolet TrailBlazer SS 2009", "Chevrolet Camaro Coupe 2012", "Chevrolet Sonic Sedan 2012",
            "Chevrolet Sonic Hatchback 2012", "Chevrolet HHR Sedan 2010", "Chevrolet Impala Sedan 2007",
            "Chevrolet Tahoe SUV 2012", "Chevrolet Traverse SUV 2012", "Chevrolet Corvette ZR1 2012",
            "Chevrolet Corvette Ron Fellows Edition Z06 2007", "Chevrolet Volt Hatchback 2012",
            "Chrysler 300 SRT-8 2010", "Chrysler Crossfire Convertible 2008", "Chrysler PT Cruiser Convertible 2008",
            "Chrysler Town and Country Minivan 2012", "Chrysler Aspen SUV 2009", "Chrysler Sebring Convertible 2010",
            "Chrysler 200 Sedan 2012", "Dodge Caliber SRT-4 2009", "Dodge Caravan Minivan 2010",
            "Dodge Ram SRT-10 Regular Cab 2006", "Dodge Ram Van Full-Size Cargo Van 2001",
            "Dodge Dakota Club Cab 2007", "Dodge Dakota Quad Cab 2010", "Dodge Challenger Coupe 2012",
            "Dodge Charger Sedan 2012", "Dodge Charger SRT-8 2009", "Dodge Magnum Wagon 2008",
            "Dodge Nitro SUV 2012", "Dodge Durango SUV 2012", "Dodge Durango SUV 2004",
            "Dodge Ram 1500 Regular Cab 2009", "FIAT 500 Abarth 2012", "FIAT 500 Convertible 2012",
            "Ferrari 458 Italia Convertible 2012", "Ferrari 458 Italia Coupe 2012", "Ferrari California Convertible 2012",
            "Ferrari FF Coupe 2012", "Fisker Karma Sedan 2012", "Ford F-150 Regular Cab 2012",
            "Ford F-150 SuperCrew 2012", "Ford F-150 SuperCab 2012", "Ford F-450 Super Duty Crew Cab 2012",
            "Ford Mustang Convertible 2007", "Ford Fiesta Sedan 2012", "Ford Club Wagon Van 2002",
            "Ford Focus Sedan 2012", "Ford Ranger SuperCab 2011", "Ford E-350 Wagon Van 2007",
            "Ford Edge SUV 2012", "Ford Fusion Sedan 2012", "Ford Flex Wagon 2012",
            "Ford GT Coupe 2006", "Ford Expedition EL SUV 2012", "Ford Taurus Sedan 2012",
            "Ford Ranger SuperCab 2005", "Ford Freestar Minivan 2007", "GMC Yukon Hybrid SUV 2012",
            "GMC Acadia SUV 2012", "GMC Canyon Extended Cab 2012", "GMC Terrain SUV 2012",
            "GMC Savana Van 2012", "GMC Envoy SUV 2006", "GMC Sierra 1500 Extended Cab 2012",
            "GMC Sierra 2500HD Crew Cab 2012", "Geo Tracker SUV 1998", "HUMMER H3T Crew Cab 2010",
            "HUMMER H2 SUT 2007", "Honda Odyssey Minivan 2012", "Honda Odyssey Minivan 2007",
            "Honda Civic Sedan 2012", "Honda Civic Coupe 2012", "Honda Accord Sedan 2012",
            "Honda Accord Coupe 2012", "Hyundai Veloster Hatchback 2012", "Hyundai Santa Fe SUV 2012",
            "Hyundai Tucson SUV 2012", "Hyundai Veracruz SUV 2012", "Hyundai Sonata Sedan 2012",
            "Hyundai Elantra Sedan 2007", "Hyundai Accent Sedan 2012", "Hyundai Genesis Sedan 2012",
            "Hyundai Sonata Sedan 2010", "Hyundai Elantra Touring Hatchback 2012", "Infiniti G Coupe 2012",
            "Infiniti QX56 SUV 2011", "Isuzu Ascender SUV 2005", "Jaguar XK XKR 2012",
            "Jeep Grand Cherokee SUV 2012", "Jeep Liberty SUV 2012", "Jeep Wrangler SUV 2012",
            "Jeep Compass SUV 2012", "Jeep Patriot SUV 2012", "Lamborghini Aventador Coupe 2012",
            "Lamborghini Gallardo LP 560-4 Coupe 2012", "Lamborghini Diablo Coupe 2001",
            "Land Rover Range Rover SUV 2012", "Land Rover LR4 SUV 2012", "Lincoln Town Car Sedan 2011",
            "Lincoln Navigator SUV 2012", "MINI Cooper Convertible 2012", "Maybach Landaulet Convertible 2012",
            "Mazda Tribute SUV 2006", "McLaren MP4-12C Coupe 2012", "Mercedes-Benz 300-Class Convertible 1993",
            "Mercedes-Benz C-Class Sedan 2012", "Mercedes-Benz SL-Class Convertible 2009",
            "Mercedes-Benz E-Class Coupe 2012", "Mercedes-Benz Sprinter Van 2012", "Mercedes-Benz SLS AMG Coupe 2012",
            "Mercedes-Benz CL-Class Coupe 2006", "Mercedes-Benz SLS AMG Convertible 2012",
            "Mitsubishi Lancer Sedan 2012", "Nissan Leaf Hatchback 2012", "Nissan NV Passenger Van 2012",
            "Nissan Juke SUV 2012", "Nissan 240SX Coupe 1998", "Nissan Murano SUV 2012",
            "Nissan GT-R Coupe 2012", "Plymouth Neon Sedan 1999", "Porsche Panamera Sedan 2012",
            "Ram C-V Cargo Van Minivan 2012", "Rolls-Royce Ghost Sedan 2012", "Rolls-Royce Phantom Sedan 2007",
            "Rolls-Royce Phantom Drophead Coupe Convertible 2012", "Saab 9-3 Sedan 2012",
            "Spyker C8 Convertible 2009", "Spyker C8 Coupe 2009", "Suzuki Aerio Sedan 2007",
            "Suzuki Kizashi Sedan 2012", "Tesla Model S Sedan 2012", "Toyota Prius Hatchback 2012",
            "Toyota Sequoia SUV 2012", "Toyota Camry Sedan 2012", "Toyota Corolla Sedan 2012",
            "Toyota 4Runner SUV 2012", "Volkswagen Golf Hatchback 2012", "Volkswagen Golf Hatchback 1991",
            "Volkswagen Beetle Hatchback 2012", "Volvo C30 Hatchback 2012", "Volvo 240 Sedan 1993",
            "Volvo XC90 SUV 2007"
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.targets = [item["label"] for item in self.hf_dataset]

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        item = self.hf_dataset[idx]
        pil_image = item["image"].convert("RGB")
        target = int(item["label"])

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return pil_image, target


class Cars:
    def __init__(self, preprocess, location=None, batch_size=32, num_workers=4, **kwargs):
        self.train_dataset = PytorchStanfordCars(location, 'train', preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PytorchStanfordCars(location, 'test', preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        self.classnames = self.train_dataset.classes
