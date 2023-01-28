//
//  Buttons.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/27/23.
//

import SwiftUI

let helper = Helpers()

struct Buttons: View {
    var body: some View {
        VStack{
            Button("Elevator 1") {
                helper.addToCSV("1")
            }.padding()
            Button("Elevator 2") {
                helper.addToCSV("2")
            }.padding()
            Button("Elevator 3") {
                helper.addToCSV("3")
            }.padding()
        }.padding()
    }
}

struct Buttons_Previews: PreviewProvider {
    static var previews: some View {
        Buttons()
    }
}
