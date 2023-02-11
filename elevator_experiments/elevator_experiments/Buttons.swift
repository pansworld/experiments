//
//  Buttons.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/27/23.
//

import SwiftUI

struct Buttons: View {
    @State var elProb :[String] = ["0.00","0.00","0.00"]
    private var excelFile = "elevator_experiments.xlsx"
    private var tableName = "Table1"
    private func updateWithHelper(){
        self.elProb[0]=helper.el1Prob
        self.elProb[1]=helper.el2Prob
        self.elProb[2]=helper.el3Prob
    }
    


    var body: some View {
        VStack{
            HStack{
                Button("Elevator 1") {
                    helper.addData(input: "1")
                    Task{
                        await helper.processTableRow(excelFile: self.excelFile, tableName: self.tableName, elevatorNo: 1)
                        updateWithHelper()
                            
                    }
                }.padding()
                Text(self.elProb[0])
            }
            HStack{
                Button("Elevator 2") {
                    helper.addData(input: "2")
                    Task{
                        await helper.processTableRow(excelFile: self.excelFile, tableName: self.tableName, elevatorNo: 2)
                        updateWithHelper()
                    }
                }.padding()
                Text(self.elProb[1])
            }
            HStack{
                Button("Elevator 3") {
                    helper.addData(input: "3")
                    Task{
                        await helper.processTableRow(excelFile: self.excelFile, tableName: self.tableName, elevatorNo: 3)
                        updateWithHelper()
                   }
                }.padding()
                Text(self.elProb[2])
            }
            Spacer()
            HStack{
                MyCustomLogin(value: self.$elProb)

            }.padding()


        }.padding()
            
    }
}

struct Buttons_Previews: PreviewProvider {
    static var previews: some View {
        Buttons()
    }
}

