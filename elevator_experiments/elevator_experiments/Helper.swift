//
//  Helper.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/27/23.
//

import Foundation

class Helpers {
    let addToCSV = addData

}

func addData(data: String) {
    let hour = Calendar.current.component(.hour, from: Date())
    let day = Calendar.current.component(.day, from: Date())
    let month = Calendar.current.component(.month, from: Date())
    let year = Calendar.current.component(.year, from: Date())

    var logFile: URL? {
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        let fileName = "symptom_data.csv"
        return documentsDirectory.appendingPathComponent(fileName)
    }
    
    guard let logFile = logFile else {
        return
    }

    guard let data = ("\(month)/\(day)/\(year),\(hour),\(data)\n").data(using: String.Encoding.utf8) else { return }

    if FileManager.default.fileExists(atPath: logFile.path) {
        if let fileHandle = try? FileHandle(forWritingTo: logFile) {
            fileHandle.seekToEndOfFile()
            fileHandle.write(data)
            fileHandle.closeFile()
        }
    } else {
        var csvText = "Symptom,Severity,Comment,Time\n"


             let newLine = "\(month)/\(day)/\(year),\(hour),\(data)\n"
             csvText.append(newLine)


         do {
            try csvText.write(to: logFile, atomically: true, encoding: String.Encoding.utf8)

         } catch {
             print("Failed to create file")
             print("\(error)")
         }
         print(logFile)
    }
}
