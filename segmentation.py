import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import ruptures  # Add this import


def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    duration = librosa.get_duration(y,sr)
    print("Duration:", duration)
    return y, sr


def compute_energy(y, frame_length, hop_length):
    energy_list = []

    for i in range(0, len(y), hop_length):
        # Extract the current frame, Square the frame samples, Sum the squared values to get the frame's energy
        frame_energy = np.sum(y[i:i + frame_length] ** 2)
        energy_list.append(frame_energy)
    energy = np.array(energy_list)
    return energy


def compute_features(y, sr, frame_length, hop_length):
    energy = compute_energy(y, frame_length, hop_length)
    #print("Energy length:", energy.shape)
    # Compute RMS (loudness)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    #print("rms length:", rms.shape)
    # Compute the Short-Time Fourier Transform (STFT)
    SpectralRepresentation = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    # Compute spectral flux
    spectral_flux = np.sqrt(np.sum(np.diff(SpectralRepresentation, axis=1)**2, axis=0))
    spectral_flux = np.append(spectral_flux, 0)  # Add one more value to match the length
    #print("flux length:", spectral_flux.shape)
    
    # Compute MFCCs (first 13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc = np.mean(mfcc, axis=0)
    #print("mfcc length:", mfcc.shape)
    # Normalize features
    energy /= np.max(energy)
    rms /= np.max(rms)
    spectral_flux /= np.max(spectral_flux)
    mfcc /= np.max(np.abs(mfcc)) 
    
    # Combine features (can adjust weights )
    combined_feature = 0.2*energy + rms + spectral_flux + mfcc
    return combined_feature, SpectralRepresentation


def slide_window(combined_feature, window_seconds):
    average_feature = []
    window_frames = int(window_seconds * sr / hop_length)
    #padding the combined feature in the end with feature value at the last index
    combined_feature = np.pad(combined_feature, (0, window_frames), 'edge')
    for i in range(len(combined_feature) - window_frames):
        window_average = np.mean(combined_feature[i:i + window_frames])
        average_feature.append(window_average)
    return average_feature

def convert_frame_to_time(average_feature, sr, hop_length, window_seconds):
    total_frame = len(average_feature)
    window_frames = int(window_seconds * sr / hop_length)   
    #print("Window frames:", window_frames)
    #time  =int(frame_number*hop_length/sr)
    slide_window_time = []
    for i in range(0,total_frame - window_frames,window_frames):
        feature_value = 0
        for j in range(i,i+window_frames):
            feature_value = feature_value+average_feature[j]
        feature_value = feature_value/window_frames
        slide_window_time.append(feature_value)
    #print("Slide window time:", slide_window_time)
    return np.array(slide_window_time)
    
def plot_sliding_window(average_feature, audio_path):
    plt.figure(figsize=(16, 8))
    # Plot the spectrogram
    plt.plot(average_feature)
    plt.xlabel('frame')
    plt.ylabel('Feature Value')
    
    current_dir = os.getcwd()
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = file_name + "_sliding_window.png"
    plt.title(file_name + ' Sliding Window')
    output_path = os.path.join(current_dir, output_filename)
    
    #plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
       
def plot_sliding_time(time_stamp_center, audio_path):
    plt.figure(figsize=(16, 8))
    # Plot the spectrogram
    plt.plot(time_stamp_center)
    plt.xlabel('time(s)')
    plt.ylabel('Feature Value')
    
    current_dir = os.getcwd()
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = file_name + "_sliding_window_time.png"
    plt.title(file_name + ' Sliding Window')
    output_path = os.path.join(current_dir, output_filename)
    
    #plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def perform_segmentation(combined_feature, threshold):
    peaks, _ = find_peaks(combined_feature, height=threshold)
    return peaks

def get_segment_boundaries(peaks, hop_length, sr):
    # Convert frame indices to timestamps in seconds
    times = librosa.frames_to_time(peaks, hop_length=hop_length, sr=sr)
    return times

def convert_frames_to_time(frames, sr, hop_length, window_seconds):
    """
    Convert frame indices to precise timestamps
    """
    return frames * window_seconds

def detect_change_points(signal, n_changepoints=5):
    """
    Detect change points in the signal using the Pelt algorithm
    Args:
        signal: 1D numpy array of the feature values
        n_changepoints: Number of change points to detect
    Returns:
        List of change point indices
    """
    # Prepare signal for ruptures
    signal = np.array(signal).reshape(-1, 1)
    
    # Create detection algorithm
    algo = ruptures.Pelt(model="rbf", min_size=10, jump=1).fit(signal)
    
    # Find change points
    change_points = algo.predict(pen=4)  # Increased penalty for fewer, more significant change points
    
    return change_points

def plot_with_change_points(signal, change_points, audio_path, sr, hop_length, window_seconds):
    """
    Plot the signal with detected change points and section annotations
    """
    plt.figure(figsize=(16, 8))
    
    # Convert frame indices to time
    time_axis = np.arange(len(signal)) * window_seconds
    change_points_time = [cp * window_seconds for cp in change_points]
    
    # Plot the original signal
    plt.plot(time_axis, signal, 'b-', label='Feature values')
    
    # Add vertical lines and annotations for change points
    change_points_time = sorted(change_points_time)  # Ensure points are in order
    sections = []
    
    # Add start point if not included
    if 0 not in change_points_time:
        sections.append(0)
    sections.extend(change_points_time)
    
    # Ensure sections are within the signal length
    sections = [s for s in sections if s <= time_axis[-1]]
    
    # Plot sections and add annotations
    for i in range(len(sections)-1):
        # Calculate midpoint for section label
        mid_point = (sections[i] + sections[i+1]) / 2
        
        # Add vertical line for section boundary
        plt.axvline(x=sections[i], color='r', linestyle='--', alpha=0.5)
        
        # Add section number
        plt.text(mid_point, plt.ylim()[1] + 0.05, f'Section {i+1}', 
                horizontalalignment='center', verticalalignment='bottom')
        
        # Add timestamp annotation with smaller font
        timestamp = f'{sections[i]:.2f}s'
        plt.annotate(timestamp, 
                    xy=(sections[i], plt.ylim()[0]),
                    xytext=(sections[i], plt.ylim()[0] - 0.1),
                    horizontalalignment='right',
                    verticalalignment='top',
                    rotation=45,
                    fontsize=8)
    
    # Add last section number and final timestamp
    if len(sections) > 1:
        plt.text((sections[-1] + time_axis[-1]) / 2, plt.ylim()[1] + 0.05, 
                f'Section {len(sections)}',
                horizontalalignment='center', verticalalignment='bottom')
        plt.axvline(x=sections[-1], color='r', linestyle='--', alpha=0.5)
        plt.annotate(f'{sections[-1]:.2f}s',
                    xy=(sections[-1], plt.ylim()[0]),
                    xytext=(sections[-1], plt.ylim()[0] - 0.1),
                    horizontalalignment='right',
                    verticalalignment='top',
                    rotation=45,
                    fontsize=8)
    
    plt.xlabel('Time (seconds)', labelpad=20)
    plt.ylabel('Feature Value', labelpad=20)
    plt.grid(True, alpha=0.3)
    
    current_dir = os.getcwd()
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = file_name + "_change_points.png"
    plt.title(file_name + ' Change Point Detection', pad=50)  # Adjust padding for the title
    
    # Add legend
    plt.legend(['Audio features', 'Section boundaries'])
    
    # Adjust layout to prevent label clipping
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the rect parameter to add more space
    
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_expanded_segments(change_points, window_seconds, total_duration, expansion=2.0):
    """
    Create expanded segments around change points with +/- expansion seconds
    Args:
        change_points: List of change points in frame indices
        window_seconds: Window size in seconds
        total_duration: Total duration of the audio in seconds
        expansion: Number of seconds to expand around change points (default 2.0)
    Returns:
        List of tuples containing (start_time, end_time) for each segment
    """
    expanded_segments = []
    change_points_time = [cp * window_seconds for cp in change_points]
    
    # Add 0 if not present
    if 0 not in change_points_time:
        change_points_time = [0] + change_points_time
    
    # Add final timestamp
    if change_points_time[-1] < total_duration:
        change_points_time.append(total_duration)
    
    # Create expanded segments
    for i in range(len(change_points_time) - 1):
        start_time = change_points_time[i]
        end_time = change_points_time[i + 1]
        
        # For first segment, don't expand start time
        if i == 0:
            expanded_start = start_time
        else:
            expanded_start = max(0, start_time - expansion)
            
        # For last segment, don't expand end time
        if i == len(change_points_time) - 2:
            expanded_end = end_time
        else:
            expanded_end = min(total_duration, end_time + expansion)
            
        expanded_segments.append((expanded_start, expanded_end))
    
    return expanded_segments

def plot_expanded_segments(signal, expanded_segments, audio_path, window_seconds):
    """
    Plot the signal with expanded segments
    """
    plt.figure(figsize=(16, 8))
    
    # Create time axis
    time_axis = np.arange(len(signal)) * window_seconds
    
    # Plot the original signal
    plt.plot(time_axis, signal, 'b-', label='Feature values')
    
    # Plot expanded segments
    for i, (start, end) in enumerate(expanded_segments):
        # Plot segment boundaries
        plt.axvline(x=start, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=end, color='r', linestyle='--', alpha=0.5)
        
        # Add segment number
        mid_point = (start + end) / 2
        plt.text(mid_point, plt.ylim()[1], f'Section {i+1}', 
                horizontalalignment='center', verticalalignment='bottom')
        
        # Add timestamps with smaller font
        plt.annotate(f'{start:.2f}s', 
                    xy=(start, plt.ylim()[0]),
                    xytext=(start, plt.ylim()[0] - 0.1),
                    horizontalalignment='right',
                    verticalalignment='top',
                    rotation=45,
                    fontsize=8)
        plt.annotate(f'{end:.2f}s',
                    xy=(end, plt.ylim()[0]),
                    xytext=(end, plt.ylim()[0] - 0.1),
                    horizontalalignment='right',
                    verticalalignment='top',
                    rotation=45,
                    fontsize=8)
        
        # Highlight expanded region with light color
        plt.axvspan(start, end, color='yellow', alpha=0.1)
    
    plt.xlabel('Time (seconds)', labelpad=20)
    plt.ylabel('Feature Value', labelpad=20)
    plt.grid(True, alpha=0.3)
    
    current_dir = os.getcwd()
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = file_name + "_expanded_segments.png"
    plt.title(file_name + ' Expanded Segments', pad=40)
    
    # Add legend
    plt.legend(['Audio features', 'Section boundaries'])
    
    # Adjust layout to prevent label clipping
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_segment_features(signal, start_idx, end_idx):
    """
    Analyze features of a segment to determine its characteristics
    Returns:
        Dictionary containing various feature metrics
    """
    segment = signal[start_idx:end_idx]
    
    features = {
        'energy': np.mean(segment),  # Average energy
        'energy_std': np.std(segment),  # Energy variation
        'energy_peak': np.max(segment),  # Peak energy
        'energy_ratio': np.max(segment) / np.mean(segment),  # Peak to average ratio
        'length': end_idx - start_idx  # Segment length
    }
    return features

def map_to_song_structure(segments, signal, window_seconds):
    """
    Map detected segments to standardized song structure following strict sequence:
    intro → drop1 (can be multiple) → bridge → drop2 (can be multiple) → outro
    Args:
        segments: List of (start_time, end_time) tuples
        signal: The feature signal
        window_seconds: Window size in seconds
    Returns:
        Dictionary mapping segment indices to song structure labels
    """
    # Convert time to indices
    segment_indices = [(int(start/window_seconds), int(end/window_seconds)) for start, end in segments]
    
    # Initialize mapping
    structure_mapping = {}
    
    # First segment is always intro
    structure_mapping[0] = 'intro'
    
    # Determine the number of segments
    num_segments = len(segments)
    
    # Check if last segment should be outro
    last_segment = segments[-1]
    outro_duration = last_segment[1] - last_segment[0]
    is_outro = False
    if outro_duration > 10:
        last_segment_start_idx = segment_indices[-1][0]
        last_segment_end_idx = segment_indices[-1][1]
        last_segment_avg_value = np.mean(signal[last_segment_start_idx:last_segment_end_idx])
        if last_segment_avg_value < 0.5:
            is_outro = True
            structure_mapping[num_segments - 1] = 'outro'
    
    # Calculate middle segments
    last_idx = num_segments - 1 if is_outro else num_segments
    middle_segments = list(range(1, last_idx))
    
    if len(middle_segments) >= 3:  # Need at least 3 segments for drop1, bridge, drop2
        # Find the middle point for bridge
        bridge_idx = len(middle_segments) // 2
        structure_mapping[bridge_idx + 1] = 'bridge'  # +1 because we start counting after intro
        
        # Assign drop1 to segments before bridge
        for i in range(1, bridge_idx + 1):
            structure_mapping[i] = 'drop1'
        
        # Assign drop2 to segments after bridge
        for i in range(bridge_idx + 2, last_idx):
            structure_mapping[i] = 'drop2'
    else:
        # Not enough segments for full structure, skip bridge
        mid_point = len(middle_segments) // 2
        for i in range(1, mid_point + 1):
            structure_mapping[i] = 'drop1'
        for i in range(mid_point + 1, last_idx):
            structure_mapping[i] = 'drop2'
    
    return structure_mapping

def compare_and_adjust_structures(segments1, mapping1, segments2, mapping2, window_seconds):
    """
    Compare two song structures and adjust if necessary.
    If one song doesn't have an outro, merge the outro of the other song into drop2.
    Args:
        segments1, segments2: List of (start_time, end_time) tuples for each song
        mapping1, mapping2: Structure mappings for each song
        window_seconds: Window size in seconds
    Returns:
        Adjusted structure mappings for both songs
    """
    # Check if either song lacks outro
    has_outro1 = 'outro' in mapping1.values()
    has_outro2 = 'outro' in mapping2.values()
    
    # If one has outro and other doesn't, merge outro into drop2
    if has_outro1 and not has_outro2:
        # Find outro in song1 and merge it into drop2
        outro_idx = [k for k, v in mapping1.items() if v == 'outro'][0]
        mapping1[outro_idx] = 'drop2'
    elif has_outro2 and not has_outro1:
        # Find outro in song2 and merge it into drop2
        outro_idx = [k for k, v in mapping2.items() if v == 'outro'][0]
        mapping2[outro_idx] = 'drop2'
    
    return mapping1, mapping2

def get_section_durations(segments, mapping):
    """
    Calculate the duration of each section type
    Args:
        segments: List of (start_time, end_time) tuples
        mapping: Structure mapping
    Returns:
        Dictionary with total duration for each section type
    """
    durations = {'intro': 0, 'drop1': 0, 'bridge': 0, 'drop2': 0, 'outro': 0}
    
    for i, (start, end) in enumerate(segments):
        if i in mapping:
            section_type = mapping[i]
            durations[section_type] += end - start
    
    return durations

def plot_song_structure(signal, segments, structure_mapping, audio_path, window_seconds):
    """
    Plot the signal with song structure labels
    """
    plt.figure(figsize=(16, 8))
    
    time_axis = np.arange(len(signal)) * window_seconds
    
    # Plot the original signal
    plt.plot(time_axis, signal, 'b-', label='Feature values')
    
    # Define colors for different sections
    section_colors = {
        'intro': 'green',
        'drop1': 'red',
        'bridge': 'purple',
        'drop2': 'orange',
        'outro': 'blue'
    }
    
    # Plot segments with structure labels
    for i, (start, end) in enumerate(segments):
        section_type = structure_mapping[i]
        color = section_colors[section_type]
        
        # Plot segment boundaries
        plt.axvline(x=start, color=color, linestyle='--', alpha=0.5)
        plt.axvline(x=end, color=color, linestyle='--', alpha=0.5)
        
        # Add section label
        mid_point = (start + end) / 2
        plt.text(mid_point, plt.ylim()[1] + 0.1, f'{section_type.upper()}', 
                horizontalalignment='center', verticalalignment='bottom',
                fontweight='bold')
        
        # Add timestamps with smaller font
        plt.annotate(f'{start:.2f}s', 
                    xy=(start, plt.ylim()[0]),
                    xytext=(start, plt.ylim()[0] - 0.1),
                    horizontalalignment='right',
                    verticalalignment='top',
                    rotation=45,
                    fontsize=8)
        plt.annotate(f'{end:.2f}s',
                    xy=(end, plt.ylim()[0]),
                    xytext=(end, plt.ylim()[0] - 0.1),
                    horizontalalignment='right',
                    verticalalignment='top',
                    rotation=45,
                    fontsize=8)
        
        # Highlight section with appropriate color
        plt.axvspan(start, end, color=color, alpha=0.1)
    
    plt.xlabel('Time (seconds)', labelpad=20)  # Add padding to the x-axis label
    plt.ylabel('Feature Value', labelpad=20)  # Add padding to the y-axis label
    plt.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.1, label=label.upper())
                      for label, color in section_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    current_dir = os.getcwd()
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = file_name + "_song_structure.png"
    plt.title(file_name + ' Song Structure Analysis', pad=40)  # Add padding to the title
    
    # Adjust layout with more space at the top
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the rect parameter to add space

    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_section_boundaries(segments, mapping):
    """
    Get the start and end times for each section type, combining consecutive segments
    Args:
        segments: List of (start_time, end_time) tuples
        mapping: Structure mapping
    Returns:
        Dictionary with overall (start, end) times for each section type
    """
    boundaries = {
        'intro': None,
        'drop1': None,
        'bridge': None,
        'drop2': None,
        'outro': None
    }
    
    # First pass to find the first and last occurrence of each section
    for i, (start, end) in enumerate(segments):
        if i in mapping:
            section_type = mapping[i]
            if boundaries[section_type] is None:
                boundaries[section_type] = [start, end]
            else:
                boundaries[section_type][1] = end  # Update end time
    
    return boundaries

def print_section_info(segments, mapping):
    """
    Print detailed information about each section including overall timings
    """
    # Get boundaries for each section
    boundaries = get_section_boundaries(segments, mapping)
    durations = get_section_durations(segments, mapping)
    
    print("\nSong Structure Analysis:")
    print("=" * 50)
    
    for section_type in ['intro', 'drop1', 'bridge', 'drop2', 'outro']:
        if boundaries[section_type] is not None:
            start, end = boundaries[section_type]
            print(f"\n{section_type.upper()}:")
            print(f"Time Range: {start:.2f}s - {end:.2f}s")
            print(f"Duration: {durations[section_type]:.2f}s")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python song_segmentation.py <audio_file> [<comparison_audio_file>]")
        sys.exit(1)

    # Process first song
    audio_path = sys.argv[1]
    y, sr = load_audio(audio_path)
    sr = 44100
    frame_length = 2048
    hop_length = 512
    
    # Compute features and detect segments for first song
    combined_feature, S = compute_features(y, sr, frame_length, hop_length)
    window_seconds = 1
    average_feature = slide_window(combined_feature, window_seconds)
    slide_time = convert_frame_to_time(average_feature, sr, hop_length, window_seconds)
    change_points = detect_change_points(slide_time)
    total_duration = len(slide_time) * window_seconds
    expanded_segments = create_expanded_segments(change_points, window_seconds, total_duration)
    
    # Map structure for first song
    structure_mapping = map_to_song_structure(expanded_segments, slide_time, window_seconds)
    print("\nSong 1 Analysis:")
    print("-" * 30)
    print("Structure mapping:", structure_mapping)
    print("Section durations:", get_section_durations(expanded_segments, structure_mapping))
    print_section_info(expanded_segments, structure_mapping)
    
    # If comparison song provided
    if len(sys.argv) > 2:
        comparison_path = sys.argv[2]
        print(f"\nProcessing comparison song: {comparison_path}")
        print("-" * 50)
        
        # Process second song
        y2, sr2 = load_audio(comparison_path)
        combined_feature2, S2 = compute_features(y2, sr2, frame_length, hop_length)
        average_feature2 = slide_window(combined_feature2, window_seconds)
        slide_time2 = convert_frame_to_time(average_feature2, sr2, hop_length, window_seconds)
        change_points2 = detect_change_points(slide_time2)
        total_duration2 = len(slide_time2) * window_seconds
        expanded_segments2 = create_expanded_segments(change_points2, window_seconds, total_duration2)
        
        # Map structure for second song
        structure_mapping2 = map_to_song_structure(expanded_segments2, slide_time2, window_seconds)
        print("\nSong 2 Analysis:")
        print("-" * 30)
        print("Structure mapping:", structure_mapping2)
        print("Section durations:", get_section_durations(expanded_segments2, structure_mapping2))
        print_section_info(expanded_segments2, structure_mapping2)
        
        # Compare and adjust structures
        adjusted_mapping1, adjusted_mapping2 = compare_and_adjust_structures(
            expanded_segments, structure_mapping,
            expanded_segments2, structure_mapping2,
            window_seconds
        )
        
        if adjusted_mapping1 != structure_mapping or adjusted_mapping2 != structure_mapping2:
            print("\nAdjusted Structures after comparison:")
            print("-" * 50)
            print("Song 1 adjusted structure:")
            print_section_info(expanded_segments, adjusted_mapping1)
            print("\nSong 2 adjusted structure:")
            print_section_info(expanded_segments2, adjusted_mapping2)
        
        # Plot both songs
        plot_song_structure(slide_time2, expanded_segments2, structure_mapping2, comparison_path, window_seconds)
    
    # Plot first song
    plot_song_structure(slide_time, expanded_segments, structure_mapping, audio_path, window_seconds)
    plot_with_change_points(slide_time, change_points, audio_path, sr, hop_length, window_seconds)