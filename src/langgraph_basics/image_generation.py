import os
import base64
from openai import OpenAI
from datetime import datetime

def generate_image_from_prompt(prompt: str, model: str = "gpt-image-1"):
    """
    Generate an image from a text prompt using OpenAI's GPT-image-1 model.
    
    Args:
        prompt (str): The text description of the image to generate
        model (str): Model to use - "gpt-image-1" (default)
    
    Returns:
        dict: Contains image_bytes, prompt, and metadata
    """
    try:
        # Initialize OpenAI client
        client = OpenAI()
        
        print(f"Generating image with prompt: '{prompt}'")
        print(f"Using model: {model}")
        
        # Generate image using GPT-image-1 (official API)
        result = client.images.generate(
            model=model,
            prompt=prompt
        )
        
        # Extract base64 image data and decode to bytes
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        # Estimated cost for GPT-image-1
        estimated_cost = 0.01
        
        result_data = {
            "image_bytes": image_bytes,
            "prompt": prompt,
            "model": model,
            "estimated_cost": estimated_cost,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Image generated successfully!")
        print(f"📊 Estimated cost: ${estimated_cost:.3f}")
        print(f"📏 Image size: {len(image_bytes)} bytes")
        
        return result_data
        
    except Exception as e:
        print(f"❌ Error generating image: {str(e)}")
        return None

def save_image_from_bytes(image_bytes: bytes, filename: str = None):
    """
    Save image bytes to local file in the project's images folder.
    
    Args:
        image_bytes (bytes): The image data as bytes
        filename (str): Optional filename. If not provided, uses timestamp
    
    Returns:
        str: Path to saved file or None if error
    """
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpt_image_1_{timestamp}.png"
        
        # Use images directory in current working directory
        images_dir = "images"
        
        # Create images directory if it doesn't exist
        os.makedirs(images_dir, exist_ok=True)
        filepath = os.path.join(images_dir, filename)
        
        # Save the image bytes to file
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        print(f"💾 Image saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving image: {str(e)}")
        return None

def generate_and_save_image(prompt: str, save_locally: bool = True, filename: str = None):
    """
    Complete workflow: generate image from prompt using GPT-image-1 and optionally save it locally.
    
    Args:
        prompt (str): Text description of the image
        save_locally (bool): Whether to save the image to local file
        filename (str): Optional filename for saved image
    
    Returns:
        dict: Complete result with image info and local path if saved
    """
    # Generate image with GPT-image-1
    result = generate_image_from_prompt(prompt)
    
    if result and save_locally:
        # Save image locally
        local_path = save_image_from_bytes(result["image_bytes"], filename)
        result["local_path"] = local_path
    
    return result

def estimate_monthly_cost(images_per_day: int):
    """
    Estimate monthly costs for GPT-image-1 usage.
    
    Args:
        images_per_day (int): Number of images generated per day
    
    Returns:
        dict: Cost breakdown
    """
    # GPT-image-1 has a fixed cost per image
    cost_per_image = 0.01
    daily_cost = images_per_day * cost_per_image
    monthly_cost = daily_cost * 30
    
    return {
        "cost_per_image": cost_per_image,
        "images_per_day": images_per_day,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost
    }

def quick_generate(prompt: str, filename: str = None):
    """
    Quick function to generate and save an image with minimal setup.
    
    Args:
        prompt (str): Text description of the image
        filename (str): Optional filename for the saved image
    
    Returns:
        str: Path to saved file or None if error
    """
    print(f"🎨 Quick generating: {prompt}")
    result = generate_and_save_image(prompt, filename=filename)
    
    if result and 'local_path' in result:
        print(f"✨ Success! Image saved to: {result['local_path']}")
        return result['local_path']
    else:
        print("❌ Failed to generate image")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Example prompts
    test_prompts = [
        "A serene lake surrounded by mountains at sunset",
        "A robot reading a book in a library", 
        "A colorful abstract painting with geometric shapes",
        "A minimalist modern kitchen with white cabinets",
        "A children's book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter"
    ]
    
    print("🎨 GPT-image-1 Generation Demo")
    print("=" * 50)
    print("💰 Pricing: $0.01 per image (1024x1024)")
    print("📁 Images saved to: ./images/ folder")
    print()
    
    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Enter custom prompt")
        print("2. Use example prompt")
        print("3. Quick generate (minimal prompts)")
        print("4. Cost calculator")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            user_prompt = input("Enter your image prompt: ").strip()
            if user_prompt:
                custom_filename = input("Custom filename (optional, press Enter to skip): ").strip()
                custom_filename = custom_filename if custom_filename else None
                
                result = generate_and_save_image(user_prompt, filename=custom_filename)
                if result:
                    print(f"\n✨ Generated image details:")
                    print(f"   Prompt: {result['prompt']}")
                    print(f"   Model: {result['model']}")
                    print(f"   Estimated cost: ${result['estimated_cost']:.3f}")
                    if 'local_path' in result:
                        print(f"   Saved to: {result['local_path']}")
        
        elif choice == "2":
            print("\nExample prompts:")
            for i, prompt in enumerate(test_prompts, 1):
                print(f"{i}. {prompt}")
            
            try:
                prompt_choice = int(input(f"Choose a prompt (1-{len(test_prompts)}): ")) - 1
                if 0 <= prompt_choice < len(test_prompts):
                    selected_prompt = test_prompts[prompt_choice]
                    
                    result = generate_and_save_image(selected_prompt)
                    if result:
                        print(f"\n✨ Generated image details:")
                        print(f"   Prompt: {result['prompt']}")
                        print(f"   Estimated cost: ${result['estimated_cost']:.3f}")
                        if 'local_path' in result:
                            print(f"   Saved to: {result['local_path']}")
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        
        elif choice == "3":
            print("\n🚀 Quick Generate Mode")
            print("Enter short prompts for quick image generation:")
            
            quick_prompts = []
            while True:
                prompt = input("Enter prompt (or 'done' to generate all): ").strip()
                if prompt.lower() == 'done':
                    break
                if prompt:
                    quick_prompts.append(prompt)
                    print(f"   Added: {prompt}")
            
            if quick_prompts:
                print(f"\n🎨 Generating {len(quick_prompts)} images...")
                for i, prompt in enumerate(quick_prompts, 1):
                    filename = f"quick_gen_{i}_{datetime.now().strftime('%H%M%S')}.png"
                    quick_generate(prompt, filename)
                    print()
        
        elif choice == "4":
            print("\n💰 Cost Calculator")
            try:
                images_per_day = int(input("How many images per day? "))
                
                costs = estimate_monthly_cost(images_per_day)
                
                print(f"\n📊 Cost Estimate for GPT-image-1:")
                print(f"   Cost per image: ${costs['cost_per_image']:.3f}")
                print(f"   Images per day: {costs['images_per_day']}")
                print(f"   Daily cost: ${costs['daily_cost']:.2f}")
                print(f"   Monthly cost: ${costs['monthly_cost']:.2f}")
                print(f"   Annual cost: ${costs['monthly_cost'] * 12:.2f}")
                
            except ValueError:
                print("Please enter valid numbers!")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("Invalid choice! Please try again.")

# Quick usage examples (uncomment to test):
# if __name__ == "__main__":
#     # Simple usage
#     quick_generate("A sunset over mountains")
#     
#     # With custom filename
#     quick_generate("A robot in a library", "my_robot.png")
#     
#     # Full workflow
#     result = generate_and_save_image("A colorful abstract painting")
#     if result:
#         print(f"Generated and saved to: {result['local_path']}")
