function WindLoadProportion()

    % Change to the correct working directory
    cd('/Users/lylakiani/Desktop/Research/power_system_project/Lylascase/zones');
    % Generate wind proportion for each zone

    % Set random seed
    rng(20);

    % Read wind buses
    zone1_wind_buses = readmatrix('zone1_wind_bus.csv');
    zone2_wind_buses = readmatrix('zone2_wind_bus.csv');
    zone3_wind_buses = readmatrix('zone3_wind_bus.csv');
    zone4_wind_buses = readmatrix('zone4_wind_bus.csv');
    zone5_wind_buses = readmatrix('zone5_wind_bus.csv');
    zone6_wind_buses = readmatrix('zone6_wind_bus.csv');
    zone7_wind_buses = readmatrix('zone7_wind_bus.csv');
    zone8_wind_buses = readmatrix('zone8_wind_bus.csv');

    % Read load buses
    zone1_load_buses = readmatrix('zone1_load_bus.csv');
    zone2_load_buses = readmatrix('zone2_load_bus.csv');
    zone3_load_buses = readmatrix('zone3_load_bus.csv');
    zone4_load_buses = readmatrix('zone4_load_bus.csv');
    zone5_load_buses = readmatrix('zone5_load_bus.csv');
    zone6_load_buses = readmatrix('zone6_load_bus.csv');
    zone7_load_buses = readmatrix('zone7_load_bus.csv');
    zone8_load_buses = readmatrix('zone8_load_bus.csv');

    % Generate wind proportion
    zone1_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone1_wind_buses));
    zone2_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone2_wind_buses));
    zone3_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone3_wind_buses));
    zone4_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone4_wind_buses));
    zone5_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone5_wind_buses));
    zone6_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone6_wind_buses));
    zone7_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone7_wind_buses));
    zone8_wind_proportion = 0.8 + (1.2 - 0.8) * rand(size(zone8_wind_buses));

    % Generate load proportion
    zone1_load_proportion = 1 + (2 - 1) * rand(size(zone1_load_buses));
    zone2_load_proportion = 1 + (2 - 1) * rand(size(zone2_load_buses));
    zone3_load_proportion = 1 + (2 - 1) * rand(size(zone3_load_buses));
    zone4_load_proportion = 1 + (2 - 1) * rand(size(zone4_load_buses));
    zone5_load_proportion = 1 + (2 - 1) * rand(size(zone5_load_buses));
    zone6_load_proportion = 1 + (2 - 1) * rand(size(zone6_load_buses));
    zone7_load_proportion = 1 + (2 - 1) * rand(size(zone7_load_buses));
    zone8_load_proportion = 1 + (2 - 1) * rand(size(zone8_load_buses));

    % Save wind proportion
    deleteIfExist('zone1_wind_proportion.csv');
    deleteIfExist('zone2_wind_proportion.csv');
    deleteIfExist('zone3_wind_proportion.csv');
    deleteIfExist('zone4_wind_proportion.csv');
    deleteIfExist('zone5_wind_proportion.csv');
    deleteIfExist('zone6_wind_proportion.csv');
    deleteIfExist('zone7_wind_proportion.csv');
    deleteIfExist('zone8_wind_proportion.csv');
    
    writematrix(zone1_wind_proportion, 'zone1_wind_proportion.csv');
    writematrix(zone2_wind_proportion, 'zone2_wind_proportion.csv');
    writematrix(zone3_wind_proportion, 'zone3_wind_proportion.csv');
    writematrix(zone4_wind_proportion, 'zone4_wind_proportion.csv');
    writematrix(zone5_wind_proportion, 'zone5_wind_proportion.csv');
    writematrix(zone6_wind_proportion, 'zone6_wind_proportion.csv');
    writematrix(zone7_wind_proportion, 'zone7_wind_proportion.csv');
    writematrix(zone8_wind_proportion, 'zone8_wind_proportion.csv');

    % Save load proportion
    deleteIfExist('zone1_load_proportion.csv');
    deleteIfExist('zone2_load_proportion.csv');
    deleteIfExist('zone3_load_proportion.csv');
    deleteIfExist('zone4_load_proportion.csv');
    deleteIfExist('zone5_load_proportion.csv');
    deleteIfExist('zone6_load_proportion.csv');
    deleteIfExist('zone7_load_proportion.csv');
    deleteIfExist('zone8_load_proportion.csv');
    
    writematrix(zone1_load_proportion, 'zone1_load_proportion.csv');
    writematrix(zone2_load_proportion, 'zone2_load_proportion.csv');
    writematrix(zone3_load_proportion, 'zone3_load_proportion.csv');
    writematrix(zone4_load_proportion, 'zone4_load_proportion.csv');
    writematrix(zone5_load_proportion, 'zone5_load_proportion.csv');
    writematrix(zone6_load_proportion, 'zone6_load_proportion.csv');
    writematrix(zone7_load_proportion, 'zone7_load_proportion.csv');
    writematrix(zone8_load_proportion, 'zone8_load_proportion.csv');
end

function deleteIfExist(filename)
    if exist(filename, 'file')
        delete(filename);
    end
end