//
//  elevator_experimentsApp.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/27/23.
//

import SwiftUI

@main
struct elevator_experimentsApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
